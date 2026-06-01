import sqlite3
import hashlib
import os
import re
from datetime import datetime

# ==========================================
# DATABASE CONFIGURATION
# ==========================================
DB_PATH = "fake_news_detector.db"


# ==========================================
# DATABASE INITIALIZE KARO
# ==========================================
def init_db():
    """
    Database aur tables banao agar exist nahi karte.
    Yeh function app start hone par call hota hai.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # --- Table 1: users ---
    # Login/Signup ke liye user accounts store karta hai
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL,
            email        TEXT    NOT NULL UNIQUE,
            password     TEXT    NOT NULL,       -- SHA-256 hashed password
            created_at   TEXT    NOT NULL,
            last_login   TEXT
        )
    """)

    # --- Table 2: detection_logs ---
    # Har ek prediction ka record store karta hai
    # user_id add kiya taake pata chale kaun sa user tha
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER,                    -- NULL = guest user
            input_type      TEXT    NOT NULL,           -- 'text' ya 'image'
            input_content   TEXT,                       -- actual text ya image filename
            is_valid_input  INTEGER NOT NULL,           -- 1=valid, 0=invalid
            validation_msg  TEXT,                       -- validation ka message
            prediction      TEXT,                       -- 'REAL', 'FAKE', ya NULL
            confidence      REAL,                       -- 0.0 to 1.0
            keywords        TEXT,                       -- extracted keywords (comma separated)
            sources_checked TEXT,                       -- scraped sources (Dawn, Soch etc.)
            timestamp       TEXT    NOT NULL,           -- datetime string
            ip_address      TEXT,                       -- user IP (optional logging)
            processing_time REAL,                       -- kitne seconds lage
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # --- Table 3: error_logs ---
    # Errors aur exceptions track karta hai
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS error_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            error_type  TEXT    NOT NULL,   -- 'OCR_ERROR', 'MODEL_ERROR', etc.
            error_msg   TEXT    NOT NULL,   -- actual error message
            input_type  TEXT,              -- 'text' ya 'image'
            timestamp   TEXT    NOT NULL
        )
    """)

    # --- Table 4: stats ---
    # Overall system stats (kitni real vs fake news detect hui)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            total_requests  INTEGER DEFAULT 0,
            total_real      INTEGER DEFAULT 0,
            total_fake      INTEGER DEFAULT 0,
            total_invalid   INTEGER DEFAULT 0,
            last_updated    TEXT
        )
    """)

    # Agar stats table empty hai toh initial row insert karo
    cursor.execute("SELECT COUNT(*) FROM stats")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO stats (total_requests, total_real, total_fake, total_invalid, last_updated)
            VALUES (0, 0, 0, 0, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

    conn.commit()
    conn.close()
    print("✅ Database initialized: fake_news_detector.db")


# ==========================================
# PASSWORD HASHING
# ==========================================
def hash_password(password):
    """
    Password ko SHA-256 se hash karta hai.
    Kabhi bhi plain text password store mat karo.
    """
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def validate_email(email):
    """Email format check karta hai."""
    return bool(re.match(r'^[^@]+@[^@]+\.[^@]+$', email))


# ==========================================
# USER: REGISTER (naya account banao)
# ==========================================
def register_user(name, email, password, confirm_password):
    """
    Naya user register karta hai.

    Returns:
        dict: { 'ok': True, 'user_id': int, 'name': str }
              ya
              { 'ok': False, 'field': str, 'msg': str }
    """
    # --- Validation ---
    name = name.strip()
    email = email.strip().lower()
    password = password.strip()

    if not name:
        return {'ok': False, 'field': 'name', 'msg': 'Please enter your full name.'}
    if not email or not validate_email(email):
        return {'ok': False, 'field': 'email', 'msg': 'Please enter a valid email address.'}
    if len(password) < 6:
        return {'ok': False, 'field': 'password', 'msg': 'Password must be at least 6 characters.'}
    if password != confirm_password:
        return {'ok': False, 'field': 'confirm', 'msg': 'Passwords do not match.'}

    # --- Database mein save karo ---
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO users (name, email, password, created_at)
            VALUES (?, ?, ?, ?)
        """, (name, email, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"✅ New user registered: {email}")
        return {'ok': True, 'user_id': user_id, 'name': name, 'email': email}

    except sqlite3.IntegrityError:
        # Email already exists
        return {'ok': False, 'field': 'email', 'msg': 'An account with this email already exists. Try signing in.'}
    except Exception as e:
        log_error("REGISTER_ERROR", str(e))
        return {'ok': False, 'field': 'email', 'msg': 'Something went wrong. Please try again.'}


# ==========================================
# USER: LOGIN
# ==========================================
def login_user(email, password):
    """
    User login check karta hai.

    Returns:
        dict: { 'ok': True, 'user_id': int, 'name': str, 'email': str }
              ya
              { 'ok': False, 'field': str, 'msg': str }
    """
    email = email.strip().lower()
    password = password.strip()

    if not email:
        return {'ok': False, 'field': 'email', 'msg': 'Please enter your email address.'}
    if not password:
        return {'ok': False, 'field': 'password', 'msg': 'Please enter your password.'}

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM users WHERE email = ? AND password = ?
        """, (email, hash_password(password)))

        user = cursor.fetchone()

        if not user:
            conn.close()
            return {'ok': False, 'field': 'email', 'msg': 'Email or password is incorrect. Please try again.'}

        # Last login time update karo
        cursor.execute("""
            UPDATE users SET last_login = ? WHERE id = ?
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user['id']))

        conn.commit()
        conn.close()

        print(f"✅ User logged in: {email}")
        return {'ok': True, 'user_id': user['id'], 'name': user['name'], 'email': user['email']}

    except Exception as e:
        log_error("LOGIN_ERROR", str(e))
        return {'ok': False, 'field': 'email', 'msg': 'Something went wrong. Please try again.'}


# ==========================================
# USER: GET BY ID
# ==========================================
def get_user_by_id(user_id):
    """Session se user ki info fetch karta hai."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, created_at, last_login FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        return None


# ==========================================
# LOG: EK DETECTION SAVE KARO
# ==========================================
def log_detection(
    input_type,
    input_content,
    is_valid_input,
    validation_msg,
    prediction=None,
    confidence=None,
    keywords=None,
    sources_checked=None,
    ip_address=None,
    processing_time=None,
    user_id=None          # ← naya parameter: logged in user ka ID
):
    """
    Ek detection result database mein save karta hai.
    app.py se yeh function call hoga har request par.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if isinstance(keywords, list):
            keywords = ", ".join(keywords)
        if isinstance(sources_checked, list):
            sources_checked = ", ".join(sources_checked)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO detection_logs
            (user_id, input_type, input_content, is_valid_input, validation_msg,
             prediction, confidence, keywords, sources_checked,
             timestamp, ip_address, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            input_type,
            input_content[:500] if input_content else None,
            1 if is_valid_input else 0,
            validation_msg,
            prediction,
            confidence,
            keywords,
            sources_checked,
            timestamp,
            ip_address,
            processing_time
        ))

        _update_stats(cursor, is_valid_input, prediction)
        conn.commit()
        conn.close()
        print(f"📝 Logged: [user={user_id}] [{input_type.upper()}] → {prediction or 'INVALID INPUT'}")

    except Exception as e:
        print(f"❌ Logging Error: {e}")
        log_error("LOG_ERROR", str(e), input_type)


# ==========================================
# LOG: ERROR SAVE KARO
# ==========================================
def log_error(error_type, error_msg, input_type=None):
    """Errors aur exceptions ko error_logs table mein save karta hai."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO error_logs (error_type, error_msg, input_type, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            error_type,
            str(error_msg)[:1000],
            input_type,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error logging failed: {e}")


# ==========================================
# STATS UPDATE (internal helper)
# ==========================================
def _update_stats(cursor, is_valid_input, prediction):
    cursor.execute("""
        UPDATE stats SET
            total_requests  = total_requests + 1,
            total_real      = total_real + CASE WHEN ? = 'REAL' THEN 1 ELSE 0 END,
            total_fake      = total_fake + CASE WHEN ? = 'FAKE' THEN 1 ELSE 0 END,
            total_invalid   = total_invalid + CASE WHEN ? = 0 THEN 1 ELSE 0 END,
            last_updated    = ?
        WHERE id = 1
    """, (
        prediction,
        prediction,
        1 if not is_valid_input else 0,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))


# ==========================================
# GET: RECENT LOGS
# ==========================================
def get_recent_logs(limit=20, user_id=None):
    """
    Recent detection logs return karta hai.
    user_id dene par sirf us user ke logs aate hain.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if user_id:
            cursor.execute("""
                SELECT * FROM detection_logs WHERE user_id = ?
                ORDER BY id DESC LIMIT ?
            """, (user_id, limit))
        else:
            cursor.execute("SELECT * FROM detection_logs ORDER BY id DESC LIMIT ?", (limit,))

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Get logs error: {e}")
        return []


# ==========================================
# GET: OVERALL STATS
# ==========================================
def get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stats WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception as e:
        print(f"❌ Get stats error: {e}")
        return {}


# ==========================================
# GET: ERROR LOGS
# ==========================================
def get_error_logs(limit=10):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM error_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Get error logs error: {e}")
        return []


# ==========================================
# APP.PY MEIN USE KARNE KA TARIKA:
# ==========================================
#
#   from database import init_db, register_user, login_user, log_detection
#
#   @app.route('/register', methods=['POST'])
#   def register():
#       data = request.get_json()
#       result = register_user(
#           name             = data['name'],
#           email            = data['email'],
#           password         = data['password'],
#           confirm_password = data['confirm']
#       )
#       if result['ok']:
#           session['user_id']   = result['user_id']
#           session['user_name'] = result['name']
#       return jsonify(result)
#
#   @app.route('/login', methods=['POST'])
#   def login():
#       data = request.get_json()
#       result = login_user(data['email'], data['password'])
#       if result['ok']:
#           session['user_id']   = result['user_id']
#           session['user_name'] = result['name']
#       return jsonify(result)
#
#   @app.route('/validate_text', methods=['POST'])
#   def validate_text():
#       user_id = session.get('user_id')   # None agar guest hai
#       # ... apna model code ...
#       log_detection(
#           input_type    = 'text',
#           input_content = text,
#           is_valid_input= True,
#           validation_msg= 'OK',
#           prediction    = 'REAL',
#           confidence    = 0.91,
#           user_id       = user_id        # ← yahan pass karo
#       )
# ==========================================
