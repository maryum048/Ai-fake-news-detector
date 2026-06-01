# VeritasAI — Fake News Detection System
> Final Year Project | AI-Powered News Verification Platform

---

## 📌 Project Overview

**VeritasAI** is an AI-powered fake news detection system that analyzes news articles and images to determine whether the content is **Real**, **Fake**, or **Uncertain**. It combines Machine Learning with live web scraping from credible Pakistani and international news sources to provide verified results.

---

## 🎯 Key Features

- **Text Analysis** — Paste any news article and get instant AI-powered verdict
- **Image Analysis** — Upload news screenshots; OCR extracts text automatically
- **Live Source Verification** — Cross-checks against Soch Fact Check, Dawn News, and BBC
- **User Authentication** — Register/login to track personal detection history
- **Detection History** — View past analyses with verdicts and confidence scores
- **Hybrid AI System** — Combines LSTM deep learning with Logistic Regression fallback

---

## 🏗️ System Architecture

```
User Input (Text / Image)
        ↓
  Input Validation
  (validation.py)
        ↓
  [Image Only] OCR
  Tesseract → Extract Text
        ↓
  ML Prediction
  LSTM Model (Primary)
  Logistic Regression (Fallback)
        ↓
  Keyword Extraction
  Web Scraping → Soch / Dawn / BBC
  (scraping.py)
        ↓
  Result Combination
  ML (65%) + Scraping (35%) for Text
  ML (55%) + Scraping (45%) for Image
        ↓
  Final Verdict + Database Log
  (database.py)
        ↓
  JSON Response → Frontend
  (index.html)
```

---

## 🤖 Machine Learning Models

### Trained Models (Kaggle Fake & Real News Dataset)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 97.0% | 97.0% | 97.0% | 97.0% |
| SVM | 96.0% | 96.0% | 96.0% | 96.0% |
| Random Forest | 93.1% | 93.1% | 93.1% | 93.0% |
| Naive Bayes | 86.4% | 86.6% | 86.4% | 86.3% |

### LSTM Architecture (Primary Model)

```
Embedding Layer     → 10,000 vocab, 128 dimensions
Bidirectional LSTM  → 64 units (forward + backward)
Dropout             → 50%
Bidirectional LSTM  → 32 units (forward + backward)
Dropout             → 50%
Dense               → 64 units, ReLU activation
Dropout             → 30%
Dense Output        → 1 unit, Sigmoid activation
```

---

## 🔍 Verification Sources

| Source | Type | Purpose |
|---|---|---|
| **Soch Fact Check** | Pakistani Fact-Checker | Match = Confirmed FAKE |
| **Dawn News** | Pakistani Newspaper (Est. 1947) | Match = Supports REAL |
| **BBC News** | International Broadcaster (Est. 1922) | Match = Supports REAL |

---

## 🗄️ Database Schema

```
users              → id, name, email, password (SHA-256), created_at, last_login
detection_logs     → id, user_id, input_type, input_content, prediction,
                     confidence, keywords, sources_checked, timestamp, processing_time
error_logs         → id, error_type, error_msg, input_type, timestamp
stats              → total_requests, total_real, total_fake, total_invalid
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3, Flask |
| **ML / Deep Learning** | TensorFlow / Keras, scikit-learn |
| **NLP Embeddings** | BERT (sentence-transformers) |
| **OCR** | Tesseract OCR, Pillow |
| **Web Scraping** | Requests, BeautifulSoup4 |
| **Database** | SQLite3 |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Fonts** | JetBrains Mono (Google Fonts) |

---

## 📁 Project Structure

```
VeritasAI/
│
├── app.py                        # Flask backend, routes, ML prediction
├── validation.py                 # Input validation, OCR pipeline
├── scraping.py                   # Keyword extraction, news source scraping
├── database.py                   # SQLite operations, user auth
│
├── templates/
│   └── index.html                # Single Page Application frontend
│
├── static/
│   └── project.jpeg              # Background image
│
├── models_for_flask/
│   ├── lstm_fake_news_detector.h5  # Trained LSTM model
│   ├── tokenizer.pkl               # Text tokenizer
│   ├── best_model.pkl              # Logistic Regression model
│   └── feature_metadata.pkl        # Label encoder + metadata
│
├── uploads/                      # Temporary image uploads (auto-deleted)
├── fake_news_detector.db         # SQLite database (auto-created)
└── requirements.txt              # Python dependencies
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on system
- Git

### Step 1 — Clone Repository

```bash
git clone https://github.com/yourusername/veritasai.git
cd veritasai
```

### Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Install Tesseract OCR

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

**Linux:**
```bash
sudo apt install tesseract-ocr
```

### Step 4 — Verify Model Files

Ensure `models_for_flask/` contains:
- `lstm_fake_news_detector.h5`
- `tokenizer.pkl`
- `best_model.pkl`
- `feature_metadata.pkl`

### Step 5 — Run Application

```bash
python app.py
```

Open browser and go to: **http://127.0.0.1:5000**

---

## 📋 Requirements

```
flask
werkzeug
tensorflow
keras
scikit-learn
sentence-transformers
numpy
pillow
pytesseract
requests
beautifulsoup4
lxml
```

---

## 🔐 Security Features

- **Password Hashing** — SHA-256 via Python hashlib (plain text never stored)
- **SQL Injection Prevention** — Parameterized queries with `?` placeholders
- **File Upload Security** — `secure_filename()` prevents path traversal attacks
- **File Type Whitelist** — Only `.jpg`, `.jpeg`, `.png` accepted
- **File Size Limit** — Maximum 10MB per upload
- **Immediate File Deletion** — Uploaded images deleted after OCR processing

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Homepage |
| POST | `/register` | Create new account |
| POST | `/login` | User login |
| POST | `/logout` | User logout |
| GET | `/api/me` | Get current session user |
| POST | `/api/analyze/text` | Analyze text input |
| POST | `/api/analyze/image` | Analyze image input |
| GET | `/api/logs` | Get detection history |
| GET | `/api/stats` | Get overall statistics |

---

## 📊 How Verdict is Determined

```
Soch Fact Check found    →  FAKE  (70-95% confidence)
Dawn / BBC found         →  REAL  (50-92% confidence)
Nothing found + ML FAKE  →  FAKE  (50-70% confidence)
Nothing found + ML REAL  →  UNCERTAIN (manual verification recommended)
Suspicious pattern found →  FAKE  (70% confidence)
```

### Suspicious Patterns Detected
`arrested` · `jailed` · `convicted` · `scandal` · `exposed` · `banned` · `resigned` · `fired`

---

## ⚠️ Known Limitations

- Trained on English dataset — Urdu news performance is limited
- Scraping depends on website availability — may return UNCERTAIN if sites are down
- OCR accuracy depends on image quality — blurry screenshots reduce accuracy
- SHA-256 hashing is not production-grade — bcrypt recommended for deployment
- Breaking news may not appear in sources yet — shows as UNCERTAIN

---

## 🔮 Future Improvements

- Urdu language model training
- Official news APIs to replace web scraping
- bcrypt password hashing
- Cross-validation for more robust accuracy measurement
- Browser extension for real-time news checking
- Mobile application using same Flask API

---

## 👩‍💻 Authors

**Final Year Project — 2025**

Built as academic research into AI-powered misinformation detection for Pakistani news ecosystem.

---

## 📄 Dataset

**Kaggle — Fake and Real News Dataset**
- Balanced dataset with equal real and fake news articles
- Real news sourced from Reuters
- Language: English
- Used for training all ML models

---

*VeritasAI — Latin for "Truth AI"*