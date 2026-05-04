# phish_model_xgbfe.py
# XGB+FE phishing URL detector (model module)
# Output: decision + score (phishing probability)

import os
import re
import math
import numpy as np
import joblib
from urllib.parse import urlparse

DEFAULT_MODEL_PATH = os.path.join("saved_models", "xgb_fe.joblib")

KEYWORDS = ["login","verify","account","update","secure","bank","confirm","signin","password","pay","paypal"]

# Utilities

def _normalize_for_parse(url: str) -> str:
    url = (url or "").strip()
    if not re.match(r'^[a-zA-Z]+://', url):
        return "http://" + url
    return url

def _safe_parse(url: str):
    try:
        p = urlparse(url)
    except ValueError:
        return "", "", "", ""
    scheme = (p.scheme or "").lower()
    host = (p.netloc or "").lower()
    host = host.split(":")[0] if host else ""
    path = (p.path or "").lower()
    query = (p.query or "").lower()
    return scheme, host, path, query

def _subdomain_count(host: str) -> int:
    parts = [x for x in (host or "").split(".") if x]
    return max(0, len(parts) - 2)

def _has_ip(host: str) -> int:
    return 1 if re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", host or "") else 0

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    # count chars
    from collections import Counter
    c = Counter(s)
    n = len(s)
    ent = 0.0
    for v in c.values():
        p = v / n
        ent -= p * math.log2(p)
    return float(ent)


# Feature extraction

def extract_features_xgbfe(url_raw: str) -> dict:
    """
    Returns a dict of 37 numeric features (19 base + 18 engineered).
    IMPORTANT: Must be consistent with Kaggle feature engineering.
    """
    url_raw = (url_raw or "").strip()
    url_for_parse = _normalize_for_parse(url_raw)
    scheme, host, path, query = _safe_parse(url_for_parse)

    # url_text: scheme removed, lowercased (like Kaggle)
    url_text = re.sub(r'^[a-zA-Z]+://', '', url_raw.lower()).strip()

    # base features (19) 
    feats = {}
    feats["https"] = 1.0 if url_raw.lower().startswith("https://") else 0.0

    feats["url_length"] = float(len(url_raw))
    feats["host_length"] = float(len(host))
    feats["path_length"] = float(len(path))
    feats["query_length"] = float(len(query))

    feats["subdomain_cnt"] = float(_subdomain_count(host))
    feats["has_ip"] = float(_has_ip(host))

    full = (host + path + "?" + query).lower()
    feats["entropy"] = float(_shannon_entropy(full))

    full_lower = full
    for k in KEYWORDS:
        feats[f"kw_{k}"] = 1.0 if k in full_lower else 0.0

    # engineered features (18) 
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

    # path_depth: number of path segments
    p2 = (path or "").strip("/")
    if p2 == "":
        feats["path_depth"] = 0.0
    else:
        feats["path_depth"] = float(p2.count("/") + 1)

    # num_query_params
    if not query:
        feats["num_query_params"] = 0.0
    else:
        feats["num_query_params"] = float(query.count("&") + 1)

    # log transforms
    feats["log_url_length"] = float(math.log1p(feats["url_length"]))
    feats["log_host_length"] = float(math.log1p(feats["host_length"]))
    feats["log_path_length"] = float(math.log1p(feats["path_length"]))
    feats["log_query_length"] = float(math.log1p(feats["query_length"]))

    return feats

# -------------------------
# Model wrapper
# -------------------------
class XGBFEPhishModel:
    """
    Wrapper that loads an exported artifact and exposes predict(url)->(decision, score).
    decision: "PHISHING" or "SAFE"
    score: phishing probability in [0,1]
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.features = None
        self.threshold_default = 0.613901  # recommended default (Precision>=0.95)
        self.threshold_alt = 0.471273      # max-F1
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}. "
                "Export your XGB+FE model to joblib and place it under saved_models/."
            )
        artifact = joblib.load(self.model_path)

        # Expect either a dict artifact or directly a model
        if isinstance(artifact, dict):
            self.model = artifact.get("model", None)
            self.features = artifact.get("features", None)
            self.threshold_default = float(artifact.get("threshold_default", self.threshold_default))
            self.threshold_alt = float(artifact.get("threshold_alt", self.threshold_alt))
        else:
            self.model = artifact

        if self.model is None or self.features is None:
            raise ValueError("Artifact must contain keys: 'model' and 'features'.")

    def predict(self, url: str, threshold: float | None = None):
        feats = extract_features_xgbfe(url)
        x = np.array([feats[f] for f in self.features], dtype=np.float32).reshape(1, -1)

        # score = P(y=1 | x) phishing probability
        score = float(self.model.predict_proba(x)[0, 1])

        thr = self.threshold_default if threshold is None else float(threshold)
        decision = "PHISHING" if score >= thr else "SAFE"
        return decision, score, thr
