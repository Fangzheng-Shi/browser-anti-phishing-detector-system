# app.py — PhishingLess local backend (XGB+FE)
# POST /check_url {"url":"..."} -> {"decision","score","threshold","model","latency_ms"}

import os
import re
import math
import time
import numpy as np
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# =========================
# Config
# =========================
PORT = int(os.environ.get("PORT", "5030"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("saved_models", "xgb_fe.joblib"))

# Default thresholds (can be overridden by env at runtime)
THRESHOLD_DEFAULT = float(os.environ.get("THRESHOLD_DEFAULT", "0.613901"))
THRESHOLD_ALT = float(os.environ.get("THRESHOLD_ALT", "0.471273"))

KEYWORDS = ["login","verify","account","update","secure","bank","confirm","signin","password","pay","paypal"]

app = Flask(__name__)
CORS(app)

# =========================
# URL normalization helpers
# =========================
def normalize_for_parse(u: str) -> str:
    u = (u or "").strip()
    if not re.match(r'^[a-zA-Z]+://', u):
        return "http://" + u
    return u

def normalize_home_slash(u: str) -> str:
    """
    Normalize homepage URLs:
    https://example.com/  -> https://example.com
    (only when path is '/' or '', and no query/fragment)
    """
    try:
        p = urlparse(u)
    except Exception:
        return u
    if (p.path == "/" or p.path == "") and p.query == "" and p.fragment == "":
        return u.rstrip("/")
    return u

def safe_parse(u: str):
    try:
        p = urlparse(u)
    except ValueError:
        return "", "", "", ""
    scheme = (p.scheme or "").lower()
    host = (p.netloc or "").lower()
    host = host.split(":")[0] if host else ""
    path = (p.path or "").lower()
    query = (p.query or "").lower()
    return scheme, host, path, query

def subdomain_count(host: str) -> int:
    parts = [x for x in (host or "").split(".") if x]
    return max(0, len(parts) - 2)

def has_ip(host: str) -> int:
    return 1 if re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", host or "") else 0

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    c = Counter(s)
    n = len(s)
    ent = 0.0
    for v in c.values():
        p = v / n
        ent -= p * math.log2(p)
    return float(ent)

# =========================
# Feature extraction (must match training pipeline)
# =========================
def extract_features_xgbfe(url_raw: str) -> dict:
    url_raw = (url_raw or "").strip()
    url_for_parse = normalize_for_parse(url_raw)
    scheme, host, path, query = safe_parse(url_for_parse)

    # url_text: scheme removed, lowercased
    url_text = re.sub(r'^[a-zA-Z]+://', '', url_raw.lower()).strip()

    feats = {}

    # base
    feats["https"] = 1.0 if url_raw.lower().startswith("https://") else 0.0
    feats["url_length"] = float(len(url_raw))
    feats["host_length"] = float(len(host))
    feats["path_length"] = float(len(path))
    feats["query_length"] = float(len(query))
    feats["subdomain_cnt"] = float(subdomain_count(host))
    feats["has_ip"] = float(has_ip(host))

    full = (host + path + "?" + query).lower()
    feats["entropy"] = float(shannon_entropy(full))
    for k in KEYWORDS:
        feats[f"kw_{k}"] = 1.0 if k in full else 0.0

    # engineered features on url_text
    u = url_text
    u_len = max(len(u), 1)

    feats["cnt_dot"] = float(u.count("."))
    feats["cnt_dash"] = float(u.count("-"))
    feats["cnt_at"] = float(u.count("@"))
    feats["cnt_qmark"] = float(u.count("?"))
    feats["cnt_equal"] = float(u.count("="))
    feats["cnt_slash"] = float(u.count("/"))
    feats["cnt_amp"] = float(u.count("&"))
    feats["cnt_percent"] = float(u.count("%"))
    feats["cnt_underscore"] = float(u.count("_"))

    digit_cnt = sum(ch.isdigit() for ch in u)
    alpha_cnt = sum(ch.isalpha() for ch in u)

    feats["digit_ratio"] = float(digit_cnt / (u_len + 1e-6))
    feats["alpha_ratio"] = float(alpha_cnt / (u_len + 1e-6))
    feats["non_alnum_ratio"] = float((u_len - digit_cnt - alpha_cnt) / (u_len + 1e-6))

    # path_depth
    p2 = (path or "").strip("/")
    feats["path_depth"] = 0.0 if p2 == "" else float(p2.count("/") + 1)

    # num_query_params
    feats["num_query_params"] = 0.0 if not query else float(query.count("&") + 1)

    # log transforms
    feats["log_url_length"] = float(math.log1p(feats["url_length"]))
    feats["log_host_length"] = float(math.log1p(feats["host_length"]))
    feats["log_path_length"] = float(math.log1p(feats["path_length"]))
    feats["log_query_length"] = float(math.log1p(feats["query_length"]))

    return feats

# =========================
# Model loading
# =========================
MODEL = None
MODEL_FEATURES = None
MODEL_THRESHOLD_DEFAULT = THRESHOLD_DEFAULT
MODEL_THRESHOLD_ALT = THRESHOLD_ALT

def load_artifact():
    global MODEL, MODEL_FEATURES, MODEL_THRESHOLD_DEFAULT, MODEL_THRESHOLD_ALT

    if not os.path.exists(MODEL_PATH):
        MODEL = None
        return False, f"Model artifact not found: {MODEL_PATH}"

    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        return False, "Artifact must be a dict with keys: model, features, threshold_default(optional), threshold_alt(optional)."

    MODEL = artifact.get("model", None)
    MODEL_FEATURES = artifact.get("features", None)
    if MODEL is None or MODEL_FEATURES is None:
        return False, "Artifact missing required keys: 'model' and/or 'features'."

    # artifact thresholds
    MODEL_THRESHOLD_DEFAULT = float(artifact.get("threshold_default", MODEL_THRESHOLD_DEFAULT))
    MODEL_THRESHOLD_ALT = float(artifact.get("threshold_alt", MODEL_THRESHOLD_ALT))

    # IMPORTANT: allow env override to always win (so THRESHOLD_DEFAULT=... python app.py works)
    MODEL_THRESHOLD_DEFAULT = float(os.environ.get("THRESHOLD_DEFAULT", str(MODEL_THRESHOLD_DEFAULT)))
    MODEL_THRESHOLD_ALT = float(os.environ.get("THRESHOLD_ALT", str(MODEL_THRESHOLD_ALT)))

    return True, "ok"

ok, msg = load_artifact()
print("[backend] load_artifact:", ok, msg)
print("[backend] model_path:", MODEL_PATH)
print("[backend] threshold_default:", MODEL_THRESHOLD_DEFAULT, "threshold_alt:", MODEL_THRESHOLD_ALT)

# =========================
# Routes
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "threshold_default": MODEL_THRESHOLD_DEFAULT,
        "threshold_alt": MODEL_THRESHOLD_ALT,
        "features_count": (len(MODEL_FEATURES) if MODEL_FEATURES else None)
    })

@app.route("/reload_model", methods=["POST"])
def reload_model():
    ok, msg = load_artifact()
    return jsonify({"ok": ok, "message": msg})

@app.route("/check_url", methods=["POST"])
def check_url():
    t0 = time.time()
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    # guard
    if url == "" or url.startswith("chrome://") or url.startswith("about:"):
        return jsonify({
            "decision": "SAFE",
            "score": 0.0,
            "threshold": MODEL_THRESHOLD_DEFAULT,
            "model": "xgb_fe",
            "latency_ms": int((time.time() - t0) * 1000)
        })

    if MODEL is None or MODEL_FEATURES is None:
        return jsonify({
            "decision": "ERROR",
            "score": 0.0,
            "threshold": MODEL_THRESHOLD_DEFAULT,
            "model": "xgb_fe",
            "error": f"Model not loaded. Put artifact at {MODEL_PATH} and call /reload_model.",
            "latency_ms": int((time.time() - t0) * 1000)
        }), 500

    # normalize homepage slash BEFORE feature extraction
    url_norm = normalize_home_slash(url)

    feats = extract_features_xgbfe(url_norm)
    x = np.array([feats.get(f, 0.0) for f in MODEL_FEATURES], dtype=np.float32).reshape(1, -1)

    score = float(MODEL.predict_proba(x)[0, 1])

    # allow client override threshold (optional)
    thr = data.get("threshold", None)
    thr = MODEL_THRESHOLD_DEFAULT if thr is None else float(thr)

    decision = "PHISHING" if score >= thr else "SAFE"

    # debug first few requests only
    app.config.setdefault("DEBUG_COUNT", 0)
    if app.config["DEBUG_COUNT"] < 8:
        print("REQ url=", url, "norm=", url_norm, "score=", score, "thr=", thr, "decision=", decision)
        print("DEBUG core =", {k: feats.get(k) for k in ["https","url_length","host_length","path_length","query_length","cnt_slash","path_depth"]})
        app.config["DEBUG_COUNT"] += 1

    return jsonify({
        "decision": decision,
        "score": score,
        "threshold": thr,
        "model": "xgb_fe",
        "latency_ms": int((time.time() - t0) * 1000)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
