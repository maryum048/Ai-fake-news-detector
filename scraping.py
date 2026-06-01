import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# ============================================================
# NEWS SOURCE INFO
# ============================================================

SOURCE_INFO = {
    "dawn": {
        "name":    "Dawn News",
        "website": "https://www.dawn.com",
        "country": "Pakistan",
        "founded": "1947",
        "type":    "Newspaper / Online News"
    },
    "bbc": {
        "name":    "BBC News",
        "website": "https://www.bbc.com/news",
        "country": "United Kingdom",
        "founded": "1922",
        "type":    "Public Broadcaster / Online News"
    },
    "soch": {
        "name":    "Soch Fact Check",
        "website": "https://sochfactcheck.com",
        "country": "Pakistan",
        "founded": "2017",
        "type":    "Fact-Checking Organization"
    }
}


# ============================================================
# STEP 1 — Extract Keywords
# ============================================================

def extract_keywords(news_text):
    """Remove stop words and return top 6 meaningful keywords."""

    stop_words = {
        'the','a','an','and','or','but','in','on','at','to','for',
        'of','with','by','is','are','was','were','has','have','this',
        'that','it','be','been','will','would','said','says','from',
        'they','their','also','into','after','before','not','new',
        'just','very','about','more','when','what','who','how','its',
        'following','prospects','semi','world','upon','over',
        'news','report','told','according','official','spokesman',
        'sources','statement','confirmed','released','revealed',
        'shows','appears','video','photo','image','today','yesterday',
        'hours','minutes','latest','breaking','update','alert',
        'because','until','against','between','through','during',
        'above','below','under','further','once','both','each',
        'other','own','same','than','here','there','where','which',
        'while','again','your','these','those','some','such','only',
        'then','even','back','down','came','went','make','made',
        'take','took','know','come','like','time','year',
        'could','should','every','people','right','think','well',
    }

    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", news_text)
    clean = re.sub(r"\s+", " ", clean).strip()
    words = clean.lower().split()

    freq = {}
    for word in words:
        if word not in stop_words and len(word) > 4:
            freq[word] = freq.get(word, 0) + 1

    seen, keywords = set(), []
    for word, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True):
        if word not in seen:
            seen.add(word)
            keywords.append(word)
        if len(keywords) == 6:
            break

    print(f"[Keywords] Extracted: {keywords}")
    return keywords


# ============================================================
# STEP 2 — Search Soch Fact Check
# ============================================================

def search_soch(keywords):
    """Search Soch Fact Check. If found, the news is likely fake."""

    try:
        response = requests.get(
            f"https://sochfactcheck.com/?s={'+'.join(keywords)}",
            headers=HEADERS, timeout=10
        )
    except requests.RequestException:
        print("[Soch] Request failed — returning empty results.")
        return []

    soup, results = BeautifulSoup(response.content, 'html.parser'), []

    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        if (
            'sochfactcheck.com/' in href
            and not href.endswith('/')
            and '@' not in href + text
            and text.lower() != 'methodology'
            and len(href) > 45 and len(text) > 15
            and not any(x in href for x in ['?', '#', '/category/', '/tag/', '/author/', '/page/'])
            and href not in [r['url'] for r in results]
        ):
            results.append({
                'headline': text, 'url': href,
                'source': 'Soch Fact Check', 'source_key': 'soch',
                'source_info': SOURCE_INFO['soch']
            })

    print(f"[Soch] Articles found: {len(results)}")
    return results


# ============================================================
# STEP 3 — Search Dawn News (Search Page + RSS Fallback)
# ============================================================

def search_dawn(keywords):
    """Search Dawn news. Uses search page first, falls back to RSS if needed."""

    required = 1 if len(keywords) <= 2 else 2
    results  = []

    # --- Try search page first ---
    try:
        query    = ' '.join(keywords[:4])
        response = requests.get(
            f"https://www.dawn.com/search?q={requests.utils.quote(query)}&sort=score",
            headers=HEADERS, timeout=12
        )
        if response.status_code == 200:
            soup     = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('article') or soup.find_all(class_=re.compile(r'story|result|item'))

            for article in articles:
                title_tag = article.find('h2') or article.find('h3') or article.find(class_=re.compile(r'title|heading'))
                link_tag  = article.find('a', href=True)
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                href  = link_tag['href'] if link_tag else ''
                if href.startswith('/'):
                    href = 'https://www.dawn.com' + href
                matched = sum(1 for kw in keywords if kw.lower() in title.lower())
                if matched >= required and href:
                    results.append({
                        'headline': title, 'url': href, 'matched': matched,
                        'source': 'Dawn News', 'source_key': 'dawn',
                        'source_info': SOURCE_INFO['dawn']
                    })
                if len(results) >= 5:
                    break

    except requests.RequestException as e:
        print(f"[Dawn] Search failed: {e}")

    # --- RSS fallback if search gave nothing ---
    if not results:
        print("[Dawn] Trying RSS fallback.")
        try:
            response = requests.get("https://www.dawn.com/feeds/latest-news", headers=HEADERS, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'lxml-xml')
                for item in soup.find_all('item'):
                    title = item.find('title')
                    link  = item.find('link')
                    if not title:
                        continue
                    title_text = title.get_text(strip=True)
                    matched    = sum(1 for kw in keywords if kw.lower() in title_text.lower())
                    if matched >= required:
                        results.append({
                            'headline': title_text,
                            'url':      link.get_text(strip=True) if link else '',
                            'matched':  matched,
                            'source':   'Dawn News (RSS)', 'source_key': 'dawn',
                            'source_info': SOURCE_INFO['dawn']
                        })
        except requests.RequestException as e:
            print(f"[Dawn] RSS also failed: {e}")

    results.sort(key=lambda x: x['matched'], reverse=True)
    print(f"[Dawn] Articles found: {len(results)}")
    return results


# ============================================================
# STEP 4 — Search BBC News (Search Page + RSS Fallback)
# ============================================================

def search_bbc(keywords):
    """Search BBC news. Uses search page first, falls back to RSS if needed."""

    required  = 1 if len(keywords) <= 2 else 2
    results   = []
    seen_urls = set()

    # --- Try search page first ---
    try:
        query    = ' '.join(keywords[:4])
        response = requests.get(
            f"https://www.bbc.com/search?q={requests.utils.quote(query)}&sektion=news",
            headers=HEADERS, timeout=12
        )
        if response.status_code == 200:
            soup  = BeautifulSoup(response.content, 'html.parser')
            items = (
                soup.find_all('li',  attrs={'data-testid': re.compile(r'search')}) or
                soup.find_all('div', attrs={'data-testid': re.compile(r'result|card')}) or
                soup.find_all('article')
            )
            for item in items:
                title_tag = item.find('h3') or item.find('h2') or item.find(attrs={'data-testid': re.compile(r'title|heading')})
                link_tag  = item.find('a', href=True)
                if not title_tag or not link_tag:
                    continue
                title = title_tag.get_text(strip=True)
                href  = link_tag['href']
                if href.startswith('/'):
                    href = 'https://www.bbc.com' + href
                if ('bbc.com' not in href and 'bbc.co.uk' not in href) or href in seen_urls:
                    continue
                matched = sum(1 for kw in keywords if kw.lower() in title.lower())
                if matched >= required:
                    seen_urls.add(href)
                    results.append({
                        'headline': title, 'url': href, 'matched': matched,
                        'source': 'BBC News', 'source_key': 'bbc',
                        'source_info': SOURCE_INFO['bbc']
                    })
                if len(results) >= 5:
                    break

    except requests.RequestException as e:
        print(f"[BBC] Search failed: {e}")

    # --- RSS fallback if search gave nothing ---
    if not results:
        print("[BBC] Trying RSS fallback.")
        rss_feeds = [
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "https://feeds.bbci.co.uk/news/world/south_asia/rss.xml",
            "https://feeds.bbci.co.uk/news/politics/rss.xml",
            "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
            "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
        ]    
        required_rss = len(keywords) if len(keywords) <= 2 else 2
        for rss_url in rss_feeds:
            try:
                response = requests.get(rss_url, headers=HEADERS, timeout=10)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.content, 'lxml-xml')
                for item in soup.find_all('item'):
                    title_tag = item.find('title')
                    link_tag  = item.find('link')
                    desc_tag  = item.find('description')
                    if not title_tag:
                        continue
                    title_text    = title_tag.get_text(strip=True)
                    link_text     = link_tag.get_text(strip=True) if link_tag else ''
                    desc_text     = desc_tag.get_text(strip=True) if desc_tag else ''
                    combined      = (title_text + " " + desc_text).lower()
                    matched       = sum(1 for kw in keywords if kw.lower() in combined)
                    title_matched = sum(1 for kw in keywords if kw.lower() in title_text.lower())
                    if matched >= required_rss and title_matched >= 1 and link_text not in seen_urls:
                        seen_urls.add(link_text)
                        results.append({
                            'headline': title_text, 'url': link_text, 'matched': matched,
                            'source': 'BBC News (RSS)', 'source_key': 'bbc',
                            'source_info': SOURCE_INFO['bbc']
                        })
            except requests.RequestException:
                continue

    results.sort(key=lambda x: x['matched'], reverse=True)
    print(f"[BBC] Articles found: {len(results)}")
    return results


# ============================================================
# STEP 5 — Calculate Confidence Score and Final Verdict
# ============================================================

def calculate_confidence(soch_results, dawn_results, bbc_results, keywords):
    """
    Combine results from all 3 sources and return final verdict.
      Soch found     → FAKE
      Dawn/BBC found → REAL
      Nothing found  → UNCERTAIN
    """

    kw_quality = min(len(keywords) / 4.0, 1.0)
    soch_count = len(soch_results)
    dawn_count = len(dawn_results)
    bbc_count  = len(bbc_results)

    def match_ratio(results):
        if not results:
            return 0.0
        return min(results[0].get('matched', 1) / max(len(keywords), 1), 1.0)

    # --- FAKE ---
    if soch_count > 0:
        confidence = min(70 + min(soch_count * 5, 20) + round(kw_quality * 10), 95)
        return {
            "verdict":    "FAKE",
            "confidence": confidence,
            "label":      "Fake News",
            "reason":     f"Found {soch_count} article(s) on Soch Fact Check — this news has been fact-checked and debunked.",
            "source_details": [
                {'headline': r['headline'], 'url': r['url'],
                 'source_name': SOURCE_INFO['soch']['name'],
                 'source_info': SOURCE_INFO['soch']}
                for r in soch_results[:2]
            ]
        }

    # --- REAL ---
    elif dawn_count > 0 or bbc_count > 0:
        dawn_boost = round(15 + match_ratio(dawn_results) * 10) if dawn_count > 0 else 0
        bbc_boost  = round(15 + match_ratio(bbc_results)  * 10) if bbc_count  > 0 else 0
        both_bonus = 10 if (dawn_count > 0 and bbc_count > 0) else 0
        confidence = min(50 + dawn_boost + bbc_boost + both_bonus + round(kw_quality * 10), 92)

        sources, source_details = [], []
        if dawn_count > 0:
            sources.append(f"Dawn ({dawn_count} article(s))")
            source_details += [
                {'headline': r['headline'], 'url': r['url'],
                 'source_name': SOURCE_INFO['dawn']['name'],
                 'source_info': SOURCE_INFO['dawn']}
                for r in dawn_results[:2]
            ]
        if bbc_count > 0:
            sources.append(f"BBC ({bbc_count} article(s))")
            source_details += [
                {'headline': r['headline'], 'url': r['url'],
                 'source_name': SOURCE_INFO['bbc']['name'],
                 'source_info': SOURCE_INFO['bbc']}
                for r in bbc_results[:2]
            ]
        return {
            "verdict":        "REAL",
            "confidence":     confidence,
            "label":          "Real News",
            "reason":         f"Verified on {' and '.join(sources)}.",
            "source_details": source_details
        }

    # --- UNCERTAIN ---
    else:
        return {
            "verdict":        "UNCERTAIN",
            "confidence":     round(kw_quality * 30),
            "label":          "Uncertain",
            "reason":         "No matching articles found on Dawn, BBC, or Soch Fact Check. "
                              "This may be breaking news, a regional story, or unverified content.",
            "source_details": []
        }


# ============================================================
# HELPER — Get source info for frontend display
# ============================================================

def get_source_info(source_key):
    """Return source info for a given key: 'dawn', 'bbc', or 'soch'."""
    return SOURCE_INFO.get(source_key.lower(), {})