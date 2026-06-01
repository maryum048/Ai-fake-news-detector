from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
import os
import time
import pickle
import numpy as np
import pytesseract
from PIL import Image
from validation import is_valid_text, validate_image_content, extract_text_from_image
from scraping import extract_keywords, search_soch, search_dawn, search_bbc, calculate_confidence

from database import (
    init_db, log_detection, log_error, get_recent_logs, get_stats,
    register_user, login_user, get_user_by_id
)

# ==========================================
# DETERMINISTIC PREDICTIONS — Set Random Seeds
# ==========================================
# Yeh ensure karta hai ki same text har baar same prediction de
import random
random.seed(42)
np.random.seed(42)

try:
    import tensorflow as tf
    tf.random.set_seed(42)
    print("✅ Random seeds set for deterministic predictions")
except:
    pass

app = Flask(__name__)
app.secret_key = "fyp_fake_news_detector_2025_secret_key"

# ------------------------------
# CONFIGURATIONS
# ------------------------------
UPLOAD_FOLDER      = "uploads"
MODEL_DIR          = "models_for_flask"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE      = 10 * 1024 * 1024  # 10MB
MAX_SEQUENCE_LEN   = 128               # feature_metadata.pkl se: max_length = 128

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tesseract path — Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ✅ DATABASE INITIALIZE
init_db()


# ==========================================
# LSTM MODEL LOAD — app start hone par ek baar
# ==========================================
LSTM_LOADED = False
LR_LOADED   = False

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    lstm_model = load_model(os.path.join(MODEL_DIR, "lstm_fake_news_detector.h5"))

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    LSTM_LOADED = True
    print("✅ LSTM model + tokenizer loaded")

except Exception as e:
    print(f"⚠️  LSTM load failed: {e}")

# Logistic Regression fallback
try:
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
        model_dict = pickle.load(f)
    lr_model = model_dict['model']

    with open(os.path.join(MODEL_DIR, "feature_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    label_encoder = meta['label_encoder']

    from sentence_transformers import SentenceTransformer
    bert_encoder = SentenceTransformer('bert-base-uncased')

    LR_LOADED = True
    print("✅ LR fallback model loaded")

except Exception as e:
    print(f"⚠️  LR fallback load failed: {e}")


# ==========================================
# LSTM PREDICTION
# ==========================================
def predict_with_lstm(text):
    """
    LSTM se text predict karta hai.
    Returns: ('REAL' or 'FAKE', confidence 0.0-1.0)
    """
    try:
        seq    = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LEN, padding='post', truncating='post')
        # training=False ensures dropout is disabled during inference (deterministic)
        proba  = lstm_model.predict(padded, verbose=0, training=False)[0]

        if len(proba) == 1:
            # Sigmoid output
            conf = float(proba[0])
            return ('REAL', round(conf, 3)) if conf >= 0.5 else ('FAKE', round(1 - conf, 3))
        else:
            # Softmax output
            idx   = int(np.argmax(proba))
            conf  = float(round(float(proba[idx]), 3))
            label = 'REAL' if idx == 1 else 'FAKE'
            return label, conf

    except Exception as e:
        log_error("LSTM_PREDICTION_ERROR", str(e), "text")
        return None, None


def predict_with_lr(text):
    """Logistic Regression fallback."""
    try:
        embedding  = bert_encoder.encode([text])
        pred_num   = lr_model.predict(embedding)[0]
        pred_proba = lr_model.predict_proba(embedding)[0]
        conf       = float(round(max(pred_proba), 3))
        label      = label_encoder.inverse_transform([pred_num])[0].upper()
        return label, conf
    except Exception as e:
        log_error("LR_PREDICTION_ERROR", str(e), "text")
        return None, None


def run_ml_prediction(text):
    """
    LSTM pehle, fallback LR.
    Returns: (label, confidence, model_name)
    """
    if LSTM_LOADED:
        label, conf = predict_with_lstm(text)
        if label:
            return label, conf, "LSTM"

    if LR_LOADED:
        label, conf = predict_with_lr(text)
        if label:
            return label, conf, "Logistic Regression"

    return None, None, None


def combine_ml_and_scraping(ml_label, ml_conf, scraping_verdict, input_type="text", input_text=""):
    """
    Combines ML result + scraping result to produce final verdict.
    - Text input : ML=0.65, Scraping=0.35
    - Image input: ML=0.55, Scraping=0.45  (OCR text is noisy)
    - UNCERTAIN scraping + ML FAKE  → directly FAKE
    - UNCERTAIN scraping + ML REAL  → Unverified (manual check needed)
    - Suspicious patterns (arrests, scandals) still flagged — death words removed
    """
    if not ml_label:
        return scraping_verdict

    # --- SUSPICIOUS PATTERN DETECTION ---
    # Death/died words removed — they appear in real news too (e.g. Asha Bhosle)
    # Only keeping patterns that are almost always fake/rumour
    SUSPICIOUS_PATTERNS = [
        'arrested', 'jailed', 'convicted', 'sentenced',
        'scandal', 'exposed', 'banned',
        'resigns', 'resigned', 'fired',
    ]
    text_lower    = input_text.lower()
    is_suspicious = any(pat in text_lower for pat in SUSPICIOUS_PATTERNS)
    if is_suspicious:
        matched_pattern = next((pat for pat in SUSPICIOUS_PATTERNS if pat in text_lower), "")
        print(f"[Suspicious] Pattern detected: '{matched_pattern}'")

    scraping_label = scraping_verdict.get('verdict', 'UNCERTAIN')
    scraping_conf  = scraping_verdict.get('confidence', 50) / 100.0

    # Weights based on input type
    if input_type == "image":
        ml_weight, scraping_weight = 0.55, 0.45
    else:
        ml_weight, scraping_weight = 0.65, 0.35

    # Convert labels to real scores (0.0 = definitely fake, 1.0 = definitely real)
    ml_real_score      = ml_conf if ml_label == 'REAL' else 1 - ml_conf
    scraping_real_score = (
        scraping_conf       if scraping_label == 'REAL' else
        1 - scraping_conf   if scraping_label == 'FAKE' else
        0.5                 # UNCERTAIN = neutral
    )

    final_real_score = (ml_real_score * ml_weight) + (scraping_real_score * scraping_weight)
    final_conf       = round(max(final_real_score, 1 - final_real_score) * 100)

    # Debug log
    print(f"   [Score] ML={ml_label}({ml_conf}) | Scraping={scraping_label} | Final={final_real_score:.3f}")

    # ============================================================
    # CASE 1: Scraping found nothing (UNCERTAIN)
    # ============================================================
    if scraping_label == 'UNCERTAIN':

        # LR/LSTM overconfidence fix
        raw_conf = round(ml_conf * 100)
        if raw_conf >= 95:
            raw_conf = 72
        elif raw_conf >= 85:
            raw_conf = 68
        elif raw_conf >= 75:
            raw_conf = 62

        # Suspicious pattern + no source found → FAKE
        if is_suspicious:
            return {
                'verdict':    'FAKE',
                'confidence': 70,
                'label':      'Fake News — Unverified Claim',
                'reason':     (
                    f"Suspicious claim detected: '{matched_pattern}'.\n"
                    f"No matching articles found on Soch, Dawn, or BBC.\n"
                    f"Unverified claims of this type are commonly fake or rumours."
                )
            }

        # ML says FAKE + no source found → FAKE
        if ml_label == 'FAKE':
            return {
                'verdict':    'FAKE',
                'confidence': max(50, raw_conf - 10),
                'label':      'Fake News — Not Verified by Any Source',
                'reason':     (
                    f"No matching articles found on Soch, Dawn, or BBC.\n"
                    f"ML model also predicts FAKE ({raw_conf}% confidence).\n"
                    f"This news could not be verified — likely false."
                )
            }

        # ML says REAL but no source → Unverified (could be real, can't confirm)
        return {
            'verdict':    'UNCERTAIN',
            'confidence': max(50, raw_conf - 10),
            'label':      'Unverified',
            'reason':     (
                f"No matching articles found on Soch, Dawn, or BBC.\n"
                f"ML model predicts REAL ({raw_conf}% confidence).\n"
                f"Could not verify against external sources — manual verification recommended."
            )
        }

    # ============================================================
    # CASE 2: Scraping found something (REAL or FAKE)
    # ============================================================
    if final_real_score >= 0.55:
        final_label  = 'REAL'
        final_reason = f"ML model ({int(ml_conf*100)}% confident) and source verification both indicate credible content."
    elif final_real_score <= 0.45:
        final_label  = 'FAKE'
        final_reason = f"ML model ({int(ml_conf*100)}% confident) and source verification flag this as potentially fake."
    else:
        # Borderline 0.45–0.55 → trust ML
        final_label  = ml_label
        final_reason = f"ML model leans {ml_label} ({int(ml_conf*100)}%). Manual verification recommended."

    return {
        'verdict':    final_label,
        'confidence': final_conf,
        'label':      'Credible' if final_label == 'REAL' else 'Fake News' if final_label == 'FAKE' else 'Uncertain',
        'reason':     final_reason
    }


# extract_text_from_image — validation.py se import ho rahi hai


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in to access this page.", "error")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


# ==========================================
# MAIN PAGE
# ==========================================
@app.route("/")
def index():
    return render_template("index.html")


# ==========================================
# REGISTER
# ==========================================
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'ok': False, 'field': 'email', 'msg': 'No data received.'}), 400
    result = register_user(
        name             = data.get('name', ''),
        email            = data.get('email', ''),
        password         = data.get('password', ''),
        confirm_password = data.get('confirm', '')
    )
    if result['ok']:
        session['user_id']    = result['user_id']
        session['user_name']  = result['name']
        session['user_email'] = result['email']
        print(f"✅ Registered: {result['email']}")
    return jsonify(result)


# ==========================================
# LOGIN
# ==========================================
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'ok': False, 'field': 'email', 'msg': 'No data received.'}), 400
    result = login_user(
        email    = data.get('email', ''),
        password = data.get('password', '')
    )
    if result['ok']:
        session['user_id']    = result['user_id']
        session['user_name']  = result['name']
        session['user_email'] = result['email']
        print(f"✅ Logged in: {result['email']}")
    return jsonify(result)


# ==========================================
# LOGOUT
# ==========================================
@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({'ok': True})


# ==========================================
# SESSION CHECK
# ==========================================
@app.route("/api/me")
def me():
    if 'user_id' in session:
        return jsonify({'ok': True, 'name': session['user_name'], 'email': session['user_email']})
    return jsonify({'ok': False}), 401


# ==========================================
# TEXT PREDICTION (Legacy stub — DUPLICATE REMOVED)
# Actual logic: /api/analyze/text  (AJAX/JSON)
# ==========================================
@app.route("/validate_text", methods=["POST"])
def validate_text():
    # Frontend AJAX /api/analyze/text use karta hai.
    # Yeh stub sirf backward-compatibility ke liye hai.
    flash("Please use the main interface to analyse text.", "info")
    return redirect(url_for('index'))


# ==========================================
# IMAGE PREDICTION (Legacy stub — DUPLICATE REMOVED)
# Actual logic: /api/analyze/image  (AJAX/JSON)
# ==========================================
@app.route("/validate_image", methods=["POST"])
def validate_image():
    # Frontend AJAX /api/analyze/image use karta hai.
    # Yeh stub sirf backward-compatibility ke liye hai.
    flash("Please use the main interface to analyse images.", "info")
    return redirect(url_for('index'))


# ==========================================
# HISTORY
# ==========================================
@app.route("/history")
def history():
    logs = get_recent_logs(limit=20, user_id=session.get('user_id'))
    return render_template("history.html", logs=logs)


# ==========================================
# LAST RESULT API
# ==========================================
@app.route("/api/last_result")
def api_last_result():
    result = session.get('last_result', {
        'verdict': 'Uncertain', 'confidence': 0,
        'label': 'Uncertain', 'reason': 'No result yet.'
    })
    return jsonify(result)


# ==========================================
# STATS & LOGS API
# ==========================================
@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())

@app.route("/api/logs")
def api_logs():
    # user_id=None dene se saare logs aate hain (guest + logged in)
    # user_id dene se sirf us user ke logs aate hain
    user_id = session.get("user_id")
    return jsonify(get_recent_logs(limit=50, user_id=user_id))


# ==========================================
# API: TEXT ANALYSIS (AJAX — JSON Response)
# ==========================================
@app.route("/api/analyze/text", methods=["POST"])
def api_analyze_text():
    start_time = time.time()
    text_value = request.form.get("text_input", "").strip()
    user_ip    = request.remote_addr
    user_id    = session.get('user_id')

    if not text_value:
        return jsonify({'ok': False, 'error': 'Please enter some text before clicking Analyse.'}), 400

    # Step 1: Input validation
    is_valid, val_msg = is_valid_text(text_value)
    if not is_valid:
        log_detection(input_type="text", input_content=text_value,
                      is_valid_input=False, validation_msg=val_msg,
                      ip_address=user_ip, processing_time=round(time.time()-start_time,3),
                      user_id=user_id)
        return jsonify({'ok': False, 'error': val_msg}), 400

    # Step 2: LSTM prediction
    ml_label, ml_conf, model_used = run_ml_prediction(text_value)
    print(f"🧠 ML Result: {ml_label} ({ml_conf}) via {model_used}")

    # Step 3: Keyword extraction + scraping
    keywords     = extract_keywords(text_value)
    print(f"🔑 Keywords: {keywords}")
    soch_results = search_soch(keywords)
    dawn_results = search_dawn(keywords)
    bbc_results  = search_bbc(keywords)
    scraping_verdict = calculate_confidence(soch_results, dawn_results, bbc_results, keywords)
    print(f"📰 Scraping Verdict: {scraping_verdict['verdict']} ({scraping_verdict['confidence']}%)")

    # Step 4: Combine LSTM + scraping
    final = combine_ml_and_scraping(ml_label, ml_conf, scraping_verdict,
                                    input_type="text", input_text=text_value)
    processing_time = round(time.time() - start_time, 3)
    print(f"✅ Final: {final['verdict']} | {final['confidence']}% | {processing_time}s")
    print(f"   Reason: {final['reason'][:80]}...")

    # Step 5: Log to database
    log_detection(
        input_type      = "text",
        input_content   = text_value,
        is_valid_input  = True,
        validation_msg  = final['reason'],
        prediction      = final['verdict'],
        confidence      = final['confidence'],
        keywords        = ", ".join(keywords),
        sources_checked = "LSTM + Soch + Dawn + BBC",
        ip_address      = user_ip,
        processing_time = processing_time,
        user_id         = user_id
    )

    # Store in session
    session['last_result'] = final

    return jsonify({
        'ok':           True,
        'verdict':      final['verdict'],
        'label':        final['label'],
        'confidence':   final['confidence'],
        'reason':       final['reason'],
        'sources_found': {
            'soch': len(soch_results),
            'dawn': len(dawn_results),
            'bbc':  len(bbc_results)
        },
        'soch_articles': soch_results[:3],
        'dawn_articles': dawn_results[:3],
        'bbc_articles':  bbc_results[:3],
        'scraping_verdict': scraping_verdict['verdict']
    }), 200


# ==========================================
# API: IMAGE ANALYSIS (AJAX — JSON Response)
# ==========================================
@app.route("/api/analyze/image", methods=["POST"])
def api_analyze_image():
    start_time = time.time()
    user_ip    = request.remote_addr
    user_id    = session.get('user_id')

    if "image_input" not in request.files:
        return jsonify({'ok': False, 'error': 'No image file uploaded.'}), 400

    image_file = request.files["image_input"]

    if image_file.filename == "":
        return jsonify({'ok': False, 'error': 'No image file selected.'}), 400

    if not allowed_file(image_file.filename):
        log_detection(input_type="image", input_content=image_file.filename,
                      is_valid_input=False, validation_msg="Invalid file type",
                      ip_address=user_ip, processing_time=round(time.time()-start_time,3),
                      user_id=user_id)
        return jsonify({'ok': False, 'error': 'Invalid file type. Only JPG, JPEG, PNG allowed.'}), 400

    image_file.seek(0, os.SEEK_END)
    file_size = image_file.tell()
    image_file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return jsonify({'ok': False, 'error': f'File too large. Maximum {MAX_FILE_SIZE//(1024*1024)}MB allowed.'}), 400

    if file_size == 0:
        return jsonify({'ok': False, 'error': 'The file is empty. Please select a valid image.'}), 400

    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        image_file.save(filepath)
        print(f"✅ File saved: {filepath}")

        # Step 1: Basic image validation
        is_valid, val_msg = validate_image_content(filepath)
        if not is_valid:
            if os.path.exists(filepath): os.remove(filepath)
            log_detection(input_type="image", input_content=filename,
                          is_valid_input=False, validation_msg=val_msg,
                          ip_address=user_ip, processing_time=round(time.time()-start_time,3),
                          user_id=user_id)
            return jsonify({'ok': False, 'error': val_msg}), 400

        # Step 2: OCR — image se text nikalo
        extracted_text = extract_text_from_image(filepath)
        if os.path.exists(filepath): os.remove(filepath)
        print(f"🗑️ File deleted: {filepath}")

        if not extracted_text:
            log_detection(input_type="image", input_content=filename,
                          is_valid_input=True, validation_msg="OCR extracted no usable text",
                          ip_address=user_ip, processing_time=round(time.time()-start_time,3),
                          user_id=user_id)
            return jsonify({'ok': False, 'error': 'Not enough readable text found in the image. Please upload a clearer news screenshot.'}), 400

        print(f"📄 OCR extracted {len(extracted_text)} chars")
        # ── OCR TEXT TERMINAL DISPLAY ──
        print("─" * 60)
        print(f"📝 OCR TEXT:\n{extracted_text}")
        print("─" * 60)
        word_count = len(extracted_text.split())
        print(f"📊 Word count: {word_count}")
        if word_count < 20:
            print("⚠️  WARNING: OCR text bohat chhota hai — ML prediction unreliable ho sakti hai")

        # Step 3: LSTM prediction on extracted text
        ml_label, ml_conf, model_used = run_ml_prediction(extracted_text)
        print(f"🧠 ML Result: {ml_label} ({ml_conf}) via {model_used}")

        # Step 4: Keyword extraction + scraping
        keywords         = extract_keywords(extracted_text)
        print(f"🔑 Keywords: {keywords}")
        soch_results     = search_soch(keywords)
        dawn_results     = search_dawn(keywords)
        bbc_results      = search_bbc(keywords)
        scraping_verdict = calculate_confidence(soch_results, dawn_results, bbc_results, keywords)
        print(f"📰 Scraping Verdict: {scraping_verdict['verdict']} ({scraping_verdict['confidence']}%)")

        # Step 5: Combine LSTM + scraping
        final = combine_ml_and_scraping(ml_label, ml_conf, scraping_verdict,
                                        input_type="image", input_text=extracted_text)

        processing_time = round(time.time() - start_time, 3)
        print(f"✅ Final: {final['verdict']} | {final['confidence']}% | {processing_time}s")
        print(f"   Reason: {final['reason'][:80]}...")

        # Step 6: Log to database
        log_detection(
            input_type      = "image",
            input_content   = filename,
            is_valid_input  = True,
            validation_msg  = final['reason'],
            prediction      = final['verdict'],
            confidence      = final['confidence'],
            keywords        = ", ".join(keywords) if keywords else None,
            sources_checked = "OCR + LSTM + Soch + Dawn + BBC",
            ip_address      = user_ip,
            processing_time = processing_time,
            user_id         = user_id
        )

        # Store in session
        session['last_result'] = final

        return jsonify({
            'ok':           True,
            'verdict':      final['verdict'],
            'label':        final['label'],
            'confidence':   final['confidence'],
            'reason':       final['reason'],
            'sources_found': {
                'soch': len(soch_results),
                'dawn': len(dawn_results),
                'bbc':  len(bbc_results)
            },
            'soch_articles': soch_results[:3],
            'dawn_articles': dawn_results[:3],
            'bbc_articles':  bbc_results[:3],
            'ocr_text':      extracted_text[:300],
            'scraping_verdict': scraping_verdict['verdict']
        }), 200

    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        if "tesseract" in error_msg.lower():
            log_error("OCR_ERROR", error_msg, "image")
            return jsonify({'ok': False, 'error': 'OCR error. Make sure Tesseract is installed correctly.'}), 500
        else:
            log_error("IMAGE_PROCESSING_ERROR", error_msg, "image")
            return jsonify({'ok': False, 'error': 'Something went wrong. Please try again.'}), 500


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING AI FAKE NEWS DETECTOR")
    print("="*60)
    print(f"🧠 LSTM Model  : {'✅ Loaded' if LSTM_LOADED else '❌ Not loaded — check models/'}")
    print(f"🔁 LR Fallback : {'✅ Ready'  if LR_LOADED  else '❌ Not loaded'}")
    print("🔍 Scraping    : Soch + Dawn + BBC")
    print("🖼️  Image       : OCR → LSTM + Scraping")
    print("🗄️  Database    : SQLite (fake_news_detector.db)")
    print("🔐 Auth        : /register /login /logout /api/me")
    print("🌐 Server      : http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)