
# ============================================================
# MMO GOV.UK DOCUMENT CLASSIFICATION PIPELINE - Usign  BART TF-IDF and Caching
# ============================================================

import os
from datetime import datetime

import re
import numpy as np
import pandas as pd
import requests
import torch

from bs4 import BeautifulSoup
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from transformers import pipeline

# Try to import TF-IDF components (used for the sentence reducer)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
    print("[INFO] scikit-learn detected → TF-IDF reducer enabled.")
except Exception:
    _HAS_SKLEARN = False
    print("[WARN] scikit-learn NOT detected → fallback reducer will be used.")


# ============================================================
# 0. CONFIG: Paths (adjust if needed)
# ============================================================
INPUT_PARQUET = (
    "/mnt/lab/unrestricted/muhammed.njie@marinemanagement.org.uk/published/"
    "mmo_published_guidance.parquet"
)
OUTPUT_PARQUET = (
    "/mnt/lab/unrestricted/muhammed.njie@marinemanagement.org.uk/published/"
    "classified_publications_tfidf_bart_clean.parquet"
)
# NOTE: Shared cache with BART-only pipeline so both reuse summaries
CACHE_PATH = (
    "/mnt/lab/unrestricted/muhammed.njie@marinemanagement.org.uk/published/"
    "govuk_summary_cache.parquet"
)

print("[INFO] Input parquet :", INPUT_PARQUET)
print("[INFO] Output parquet:", OUTPUT_PARQUET)
print("[INFO] Cache parquet :", CACHE_PATH)


# ============================================================
# 1. MMO Domain Lexicon (Short labels only)
# ============================================================
mmo_lexicon = {
    "Compliance": "compliance enforcement offences fines inspection illegal logbooks catch record",
    "FVL": "fishing vessel licence under10 over10 registration capacity permit",
    "Marine Planning": "marine planning regional plans seascape spatial plan spp",
    "IIU": "illegal unreported unregulated iuu catch certificate processing statement storage export",
    "SIA": "single issuance authority foreign vessel permit eu waters authorisation",
    "Planning Team": "coastal concordat plan development workshops decision making",
    "Conservative team": "protected area mpa mcz wildlife byelaw conservation species",
    "OMT": "operations coastal daily management monitoring enforcement",
    "IVMS": "ivms vessel monitoring gps tracking under12 system",
    "Corporate": "strategy privacy transparency annual reports corporate risk",
    "Stats": "statistics datasets landings analysis quality assessment",
    "Comms": "communications newsletter press announcement media",
    "Funding": "funding grants seafood fund emff mff",
    "Global Marine Team": "blue belt overseas territory marine conservation international",
    "FMT": "fisheries management team quota landing obligation discard ban",
    "FDST": "data systems ers logbooks sales notes data collection",
    "Marine Licensing": "licence dredging construction disposal impact assessment"
}

candidate_labels = list(mmo_lexicon.keys())
print("[INFO] Using short candidate labels:", candidate_labels)


# ============================================================
# 2. GOV.UK SCRAPER
# ============================================================
def get_gov_summary(url, max_chars=2000):
    print(f"[SCRAPER] Fetching URL: {url}")

    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        print("[SCRAPER] Page fetched successfully.")
    except Exception:
        print("[SCRAPER][WARN] Failed → empty summary returned.")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    NOISE = [
        "nav", "header", "footer", "aside",
        ".gem-c-breadcrumbs", ".gem-c-related-navigation",
        ".app-c-related-navigation", ".gem-c-metadata",
        ".gem-c-publisher-metadata", ".gem-c-contextual-menu",
        ".gem-c-title", ".govuk-breadcrumbs", ".govuk-footer",
        ".govuk-header", ".gem-c-layout-super-navigation-header",
        ".gem-c-layout-super-navigation-footer",
        ".gem-c-topic-list", ".gem-c-document-list", ".gem-c-figure",
        "#related-content", "#full-page-navigation",
        ".support-links", ".govuk-accordion", ".app-c-contents-list",
        "#global-cookie-message"
    ]

    print("[SCRAPER] Removing GOV.UK boilerplate...")
    for sel in NOISE:
        for tag in soup.select(sel):
            tag.decompose()

    COOKIE_PHRASES = [
        "cookies on gov.uk", "accept additional cookies",
        "reject additional cookies", "view cookies"
    ]
    for node in soup.find_all(string=True):
        if any(p in node.lower() for p in COOKIE_PHRASES):
            if node.parent:
                node.parent.decompose()

    main = soup.select_one(".govuk-main-wrapper, .govuk-grid-column-two-thirds") or soup.body
    if not main:
        print("[SCRAPER][WARN] No main content found.")
        return ""

    for a in main.find_all("a"):
        a.unwrap()

    ALLOWED = ["h1", "h2", "h3", "h4", "p", "li"]
    SKIP = ["is this page useful", "help us improve", "sign up"]

    chunks = []
    print("[SCRAPER] Extracting clean text blocks...")
    for tag in main.find_all(ALLOWED):
        text = tag.get_text(" ", strip=True)
        if len(text) > 5 and not any(s in text.lower() for s in SKIP):
            chunks.append(text)

    seen, cleaned = set(), []
    for c in chunks:
        if c not in seen:
            cleaned.append(c)
            seen.add(c)

    full = " ".join(cleaned)
    return full[:max_chars]


# ============================================================
# 3. SUMMARY CACHE (URL → Summary)
# ============================================================
def _dbfs_exists(path: str) -> bool:
    """Check existence of a DBFS file by probing /dbfs mirror."""
    return os.path.exists(f"/dbfs{path}")

def load_cache(cache_path: str) -> pd.DataFrame:
    """Load existing cache parquet, or return empty cache DataFrame."""
    if _dbfs_exists(cache_path):
        print("[CACHE] Loading existing GOV.UK summary cache...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"[CACHE][WARN] Failed to read cache ({e}). Starting fresh.")
            return pd.DataFrame(columns=["URL", "Summary", "Last_Scraped"])
    else:
        print("[CACHE] No existing cache found. Creating new cache DataFrame.")
        return pd.DataFrame(columns=["URL", "Summary", "Last_Scraped"])

# Load cache and build in-memory dict
cache_df = load_cache(CACHE_PATH)
cache = dict(zip(cache_df["URL"], cache_df["Summary"]))
print(f"[CACHE] Cached URL count: {len(cache)}")

def get_gov_summary_cached(url, max_chars=2000):
    """Return cached summary for URL when available; otherwise scrape and cache."""
    if not isinstance(url, str) or not url.strip():
        return ""

    if url in cache:
        print(f"[CACHE HIT] {url}")
        return cache[url]

    # Cache miss → scrape, then cache
    print(f"[CACHE MISS] {url} → scraping")
    summary = get_gov_summary(url, max_chars=max_chars)
    cache[url] = summary
    return summary


# ============================================================
# 4. Load MMO Parquet, scrape with cache, and build Text_For_Model
# ============================================================
print("[INFO] Loading MMO parquet...")
df = spark.read.parquet(INPUT_PARQUET)

pdf = df.toPandas()
print("[INFO] Total documents:", len(pdf))

if "URL" in pdf.columns:
    print("[INFO] Scraping GOV.UK pages with cache...")
    pdf["Summary"] = pdf["URL"].apply(get_gov_summary_cached)
else:
    print("[WARN] URL column missing → empty summaries used.")
    pdf["Summary"] = ""

# Build Raw_Text then reduce to Text_For_Model via TF-IDF sentence selection
pdf["Raw_Text"] = (
    pdf.get("Title", pd.Series([""] * len(pdf))).fillna("").astype(str) + " " +
    pdf["Summary"].fillna("").astype(str)
).str.strip()


# ============================================================
# 5. TF‑IDF Sentence Reducer
# ============================================================
_SENT_SPLIT_RE = re.compile(r'[.!?]\s+(?=[A-Z])')

def split_sentences(text):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.strip()) >= 25]

    seen, out = set(), []
    for s in sents:
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out

def _clip(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]

def tfidf_reduce(text, max_sentences=8, max_chars=1500, diversity=0.7):
    sents = split_sentences(text)
    if not sents:
        return _clip((text or "").strip(), max_chars)

    if not _HAS_SKLEARN:
        return _clip(" ".join(sents[:max_sentences]), max_chars)

    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform(sents).toarray()

    centroid = X.mean(axis=0, keepdims=True)
    rel = cosine_similarity(X, centroid).ravel()

    simmat = cosine_similarity(X)
    k = min(max_sentences, len(sents))

    selected = [int(np.argmax(rel))]
    candidates = set(range(len(sents))) - set(selected)

    while len(selected) < k and candidates:
        best_score = -1e9
        best_idx = None
        for i in candidates:
            diversity_pen = max(simmat[i, j] for j in selected)
            score = diversity * rel[i] - (1 - diversity) * diversity_pen
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(best_idx)
        candidates.remove(best_idx)

    summary = " ".join(sents[i] for i in sorted(selected))
    return _clip(summary, max_chars)

print("[INFO] Applying TF‑IDF reducer to model text...")
pdf["Text_For_Model"] = pdf["Raw_Text"].apply(
    lambda t: tfidf_reduce(t, max_sentences=8, max_chars=1500, diversity=0.7)
)

# Persist the updated cache now (include a timestamp)
print("[CACHE] Persisting updated cache to Parquet...")
updated_cache_df = pd.DataFrame({
    "URL": list(cache.keys()),
    "Summary": list(cache.values()),
    "Last_Scraped": datetime.utcnow().isoformat()
})
updated_cache_df = updated_cache_df.drop_duplicates(subset=["URL"])
updated_cache_df.to_parquet(CACHE_PATH, index=False)
print("[CACHE] Cache saved:", CACHE_PATH)

# Back to Spark
spark_df = spark.createDataFrame(pdf.drop(columns=["Raw_Text"]))


# ============================================================
# 6. BART Classifier (Structured Outputs + Review Flags + Verbose Logs)
# ============================================================
MIN_CONF = 0.30
MARGIN_THRESHOLD = 0.05

schema = StructType([
    StructField("bart_team", StringType(), True),
    StructField("bart_score", FloatType(), True),
    StructField("bart_margin", FloatType(), True),
    StructField("bart_final", StringType(), True),
    StructField("review_flag", StringType(), True),
])

_bart_singleton = None

@pandas_udf(schema)
def classify_bart_udf(texts: pd.Series) -> pd.DataFrame:

    global _bart_singleton

    if _bart_singleton is None:
        print("\n[INFO] Loading BART MNLI model for this Spark worker...")
        device = 0 if torch.cuda.is_available() else -1
        print(f"[INFO] Device used: {device}")

        _bart_singleton = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device
        )
        print("[INFO] BART initialised.")

    clf = _bart_singleton
    rows = []

    print(f"\n[INFO] Starting classification of batch size: {len(texts)}")

    for idx, text in enumerate(texts):
        print(f"\n===== ROW {idx+1} START =====")

        if not text or len(text.strip()) < 5:
            print("[WARN] Empty text → forced Unclassified + REVIEW_EMPTY")
            rows.append(("Unclassified", 0.0, 0.0, "Unclassified", "REVIEW_EMPTY"))
            continue

        hypothesis_template = "This GOV.UK document relates to {}."

        print("[INFO] Running BART prediction...")
        out = clf(
            sequences=text,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
            truncation=True
        )

        labels = out["labels"]
        scores = out["scores"]

        bart_team = labels[0]
        bart_score = float(scores[0])
        bart_margin = float(scores[0] - scores[1]) if len(scores) > 1 else bart_score

        print(f"[INFO] Top label: {bart_team}")
        print(f"[INFO] Score: {bart_score:.4f}, Margin: {bart_margin:.4f}")

        if bart_score >= MIN_CONF and bart_margin >= MARGIN_THRESHOLD:
            bart_final = bart_team
            print("[INFO] Decision: ACCEPTED")
        else:
            bart_final = "Unclassified"
            print("[INFO] Decision: REJECTED → Unclassified")

        if bart_score < 0.20:
            review_flag = "REVIEW_LOW_SCORE"
        elif bart_score < MIN_CONF:
            review_flag = "REVIEW_SCORE_BELOW_THRESHOLD"
        elif bart_margin < 0.02:
            review_flag = "REVIEW_LOW_MARGIN"
        else:
            review_flag = "OK"

        print(f"[INFO] Review Flag: {review_flag}")

        rows.append((bart_team, bart_score, bart_margin, bart_final, review_flag))

        print(f"===== ROW {idx+1} END =====")

    print("[INFO] Batch classification complete.")
    return pd.DataFrame(rows, columns=[
        "bart_team", "bart_score", "bart_margin", "bart_final", "review_flag"
    ])


# ============================================================
# 7. EXECUTE CLASSIFIER + SAVE RESULTS
# ============================================================
print("\n[INFO] Running full document classification...")

final_df = (
    spark_df
    .withColumn("bart_struct", classify_bart_udf(col("Text_For_Model")))
    .select(
        "*",
        col("bart_struct.bart_team"),
        col("bart_struct.bart_score"),
        col("bart_struct.bart_margin"),
        col("bart_struct.bart_final"),
        col("bart_struct.review_flag")
    )
    .drop("bart_struct")
)

print(f"[INFO] Writing final results to {OUTPUT_PARQUET} ...")
final_df.write.mode("overwrite").parquet(OUTPUT_PARQUET)
print("[INFO] Pipeline completed successfully!")
