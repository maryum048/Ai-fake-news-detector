from PIL import Image
import pytesseract
import re
import os

# ==========================================
# TESSERACT CONFIGURATION
# ==========================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

TESSERACT_AVAILABLE = False
try:
    if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
        print("✅ Tesseract OCR is available")
    else:
        print(f"⚠️ Tesseract not found at: {pytesseract.pytesseract.tesseract_cmd}")
except Exception as e:
    print(f"⚠️ Tesseract warning: {e}")

# ==========================================
# IMAGE TYPE CHECKER
# OCR-based text density check — no PyTorch needed
# ==========================================
CLIP_AVAILABLE = True
clip_model     = None
clip_processor = None
print("✅ Image type checker ready (OCR-based)")

# ==========================================
# SHARED VOCABULARY — EXPANDED
# Used by: is_valid_text + clean_ocr_text + is_news_image_clip
# ==========================================
_KNOWN_WORDS = {
    # Common English words
    'the','be','to','of','and','a','in','that','have','it','for','not','on','with',
    'he','as','you','do','at','this','but','his','by','from','they','we','say',
    'her','she','or','an','will','my','one','all','would','there','their','what',
    'so','up','out','if','about','who','get','which','go','me','when','make','can',
    'like','time','no','just','him','know','take','people','into','year','your',
    'good','some','could','them','see','other','than','then','now','look','only',
    'come','its','over','think','also','back','after','use','two','how','our','work',
    'first','well','way','even','new','want','because','any','these','give','day',
    'most','us','were','been','those','since','years','three','four','five','last',
    'next','where','more','told','asked','added','former','while','during','before',
    'after','against','under','federal','provincial','are','was','has','had','will',
    'would','could','should','said','says','is','are','was','were','has','have',
    # News-specific words
    'pakistan','lahore','karachi','islamabad','said','news','police','incident',
    'minister','party','government','president','prime','court','case','report',
    'national','political','statement','chief','commission','responsible','model',
    'town','killings','justice','conference','resign','held','fully','crimes',
    'conceal','confuse','planners','awami','beyond','doubt','attack','kill','die',
    'army','military','election','vote','law','act','anti','terror','terrorism',
    'murder','arrested','accused','tehreek','sharif','shahbaz','qadri','baqar',
    'naqvi','sahiwal','bragged','humiliation','ignominy','ambassador','ombudsman',
    'promotions','intimidated','forensic','concealed','investigation','india',
    'railways','safety','accidents','fighter','aircraft','crashes','mumbai',
    'earnings','minister','revenue','production','freight','passenger','train',
    'read','published','today','yesterday','latest','breaking','update','alert',
    'confirmed','sources','according','spokesman','official','spokesperson','told',
    'video','photo','image','shows','appears','revealed','exposed','released',
    'hours','minutes','seconds','ago','morning','evening','night','deadline',
    'killed','injured','wounded','died','dead','deaths','survived','missing',
    'hospital','clinic','medical','health','patient','doctor','nurse','disease',
    'protest','rally','march','demonstration','gathering','crowd','activists',
    'group','organization','association','society','committee','board','council',
    'agreement','contract','deal','proposal','plan','project','scheme','initiative',
    'international','global','regional','local','foreign','domestic','internal',
    'development','infrastructure','construction','building','development','project',
    'economic','financial','business','commercial','trade','market','investment',
    'technology','digital','internet','online','cyber','software','hardware',
    'education','school','university','college','student','teacher','academic',
    'sports','match','game','team','player','coach','victory','defeat','champion',
    'weather','climate','rain','snow','temperature','forecast','storm','flood',
    'crime','criminal','police','investigation','evidence','arrest','trial','court',
    'business','company','corporation','industry','sector','entrepreneur','startup',
    # Urdu/Hindi common words
    'hai','hain','tha','thi','ka','ki','ke','ne','ko','se','mein','aur','ya',
    'jo','kia','nahi','yeh','wah','koi','sab','kuch'
}


# ==========================================
# TEXT VALIDATION
# ==========================================
def is_valid_text(text):
    """Validate user-submitted text (slightly relaxed)"""
    text = text.strip()
    if len(text) < 30:
        return False, "Text is too short. Please enter at least 30 characters."
    if len(text) > 10000:
        return False, "Text is too long. Maximum 10,000 characters allowed."
    
    letters = sum(c.isalpha() for c in text)
    # Relaxed from 0.40 to 0.35
    if letters / max(1, len(text)) < 0.35:
        return False, "Text contains too many special characters."
    
    words = text.split()
    if len(words) < 5:
        return False, "Please enter at least 5 words."
    
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) > 10:
        if sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.85:
            return False, "Please do not write in all caps."
    
    if re.search(r'(.)\1{6,}', text):
        return False, "Text contains too many repeated characters."
    
    # Relaxed: require at least 1 known word if text > 10 words (was: 0)
    word_list = re.findall(r'\b\w+\b', text.lower())
    if len(word_list) > 10 and sum(1 for w in word_list if w in _KNOWN_WORDS) == 0:
        return False, "Text does not seem meaningful. Please enter a real news article."
    
    return True, "Text is valid."


# ==========================================
# IMAGE PREPROCESSOR  ← MUST be before is_news_image_clip
# ==========================================
def preprocess_image_for_ocr(img):
    """
    Image ko OCR ke liye enhance karta hai:
    - Grayscale conversion
    - Contrast + sharpness boost
    - Resize agar bohat chhoti ho
    """
    from PIL import ImageEnhance
    img = img.convert("L")
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    return img.convert("RGB")


# ==========================================
# OCR TEXT CLEANER  ← MUST be before is_news_image_clip
# ==========================================
def clean_ocr_text(text):
    """
    OCR noise remove karta hai (RELAXED):
    1. Remove obvious garbage only (pure symbols, headers)
    2. Keep all lines with 2+ words or sufficient alphabetic content
    3. Only remove if <20% alphabetic (symbol/digit noise)
    """
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove pure symbol lines
        if len(re.findall(r'[a-zA-Z]', line)) == 0:
            continue
        
        # Remove if >70% non-ASCII (not >40%)
        non_ascii = sum(1 for c in line if ord(c) > 127)
        if non_ascii / max(1, len(line)) > 0.7:
            continue
        
        # Remove if <20% alphabetic content
        alpha = sum(c.isalpha() for c in line)
        if len(line) > 0 and alpha / len(line) < 0.20:
            continue
        
        # Keep all-caps lines UNLESS they're very short headers (<=8 chars)
        if len(line) <= 8 and line.isupper() and line.isalpha():
            continue
        
        # Keep lines with 2+ words
        word_count = len(re.findall(r'\b[a-zA-Z]{2,}\b', line))
        if word_count >= 2:
            clean_lines.append(line)
        # Keep longer lines even with 1 word (might be headline or URL)
        elif len(line) > 15:
            clean_lines.append(line)
    
    cleaned = ' '.join(clean_lines)
    # Remove excessive special characters but keep some structure
    cleaned = re.sub(r'[|_]{2,}', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ==========================================
# IMAGE TYPE CHECK — OCR Text Density Method
# preprocess_image_for_ocr aur clean_ocr_text
# dono upar define hain — ab safely call ho sakti hain
# ==========================================
def is_news_image_clip(image_path):
    """
    News image detector — OCR text density approach (RELAXED).
    >= 8 words   → likely news image
     3-7 words   → quality check
    < 3 words    → not a news image
    """
    if not TESSERACT_AVAILABLE:
        return True, "OCR not available — skipping image type check."

    try:
        img = Image.open(image_path).convert("RGB")

        # Preprocess first — same enhancement as extract_text_from_image
        img = preprocess_image_for_ocr(img)

        # Quick OCR pass
        raw_text = pytesseract.image_to_string(
            img, lang='eng', config='--oem 3 --psm 3'
        ).strip()

        words = re.findall(r'\b[a-zA-Z]{3,}\b', raw_text)
        word_count = len(words)
        print(f"🔍 Image type check: {word_count} words detected")

        if word_count >= 8:
            return True, f"News image confirmed ({word_count} words detected)."
        elif word_count >= 3:
            known_hits = sum(1 for w in words if w.lower() in _KNOWN_WORDS)
            ratio = known_hits / word_count if word_count > 0 else 0
            # Relaxed requirement: only 10% known words needed
            if ratio >= 0.10 or word_count >= 5:
                return True, f"Possible news image ({word_count} words detected)."
            else:
                # Even stricter rejection - give them a chance if >= 3 words
                return True, f"Proceeding with OCR ({word_count} words found, will validate after extraction)."
        else:
            return False, (
                f"Too little text detected in image ({word_count} words). "
                "Please upload a clearer news screenshot."
            )
    except Exception as e:
        print(f"⚠️ Image type check error: {e} — proceeding anyway")
        return True, "Image type check skipped due to error."


# ==========================================
# OCR — IMAGE SE TEXT NIKALO
# ==========================================
def extract_text_from_image(image_path):
    """
    3 PSM modes try karta hai, best result use karta hai.
    preprocess + clean_ocr_text automatically apply hoti hain.
    """
    if not TESSERACT_AVAILABLE:
        print("⚠️ Tesseract not available")
        return ""
    try:
        img = Image.open(image_path).convert("RGB")
        img_processed = preprocess_image_for_ocr(img)

        text_psm3 = pytesseract.image_to_string(
            img_processed, lang='eng', config='--oem 3 --psm 3'
        ).strip()
        text_psm4 = pytesseract.image_to_string(
            img_processed, lang='eng', config='--oem 3 --psm 4'
        ).strip()
        text_psm6 = pytesseract.image_to_string(
            img_processed, lang='eng', config='--oem 3 --psm 6'
        ).strip()

        best_text = max([text_psm3, text_psm4, text_psm6], key=len)
        best_text = clean_ocr_text(best_text)
        print(f"🧹 OCR cleaned text: {len(best_text)} chars")
        return best_text
    except Exception as e:
        print(f"⚠️ OCR error: {e}")
        return ""


# ==========================================
# IMAGE TEXT VALIDATION
# ==========================================
def is_valid_image_text(extracted_text):
    """RELAXED validation - allow more content through"""
    # Reduced from 20 to 10 characters
    if len(extracted_text) < 10:
        return False, "No readable text found in image."
    
    words = extracted_text.split()
    # Reduced from 4 to 2 words
    if len(words) < 2:
        return False, f"Too little text found ({len(words)} words). Please upload a clearer image."
    
    letters = sum(c.isalpha() for c in extracted_text)
    # Reduced from 0.30 to 0.20
    if letters / max(1, len(extracted_text)) < 0.20:
        return False, "Image text is mostly numbers or symbols."
    
    real_words = re.findall(r'\b[a-zA-Z]{2,}\b', extracted_text)
    # Reduced from 3 to 1 word
    if len(real_words) < 1:
        return False, "Could not extract meaningful text. Please upload a clearer image."
    
    return True, "Image text is valid."


# ==========================================
# MAIN PIPELINE — app.py yahi call karta hai
# ==========================================
def validate_image_content(image_path):
    """
    Pipeline:
      1. CLIP/OCR check — news image hai?
      2. OCR — text nikalo
      3. Text validate karo
      4. (True, extracted_text) ya (False, error_message)
    """
    is_news, clip_msg = is_news_image_clip(image_path)
    if not is_news:
        return False, clip_msg

    if not TESSERACT_AVAILABLE:
        return False, (
            "Tesseract OCR is not installed. "
            "Please install it from: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    extracted_text = extract_text_from_image(image_path)

    if not extracted_text or len(extracted_text.strip()) < 20:
        return False, (
            "Could not extract readable text from this image. "
            "Please upload a clearer news screenshot with visible text."
        )

    is_valid, text_msg = is_valid_image_text(extracted_text)
    if not is_valid:
        return False, text_msg

    return True, extracted_text
    