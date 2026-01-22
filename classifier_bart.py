
# ============================================================
# MMO DOCUMENT CLASSIFICATION PIPELINE (BART‑ONLY) + CACHING
# - Short labels
# - Structured outputs
# - Relaxed thresholds
# - Review flags
# - Verbose print statements
# - Persistent GOV.UK URL→Summary cache
# ============================================================

import os
from datetime import datetime

import re
import pandas as pd
import requests
import torch

from bs4 import BeautifulSoup
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from transformers import pipeline

# ============================================================
# 0. CONFIG: Paths (adjust if you prefer a different location)
# ============================================================
INPUT_PARQUET = (
    "/mnt/lab/unrestricted/muhammed.njie@marinemanagement.org.uk/published/"
    "mmo_published_guidance.parquet"
)
OUTPUT_PARQUET = (
    "/mnt/lab/unrestricted/muhammed.njie@marinemanagement.org.uk/published/"
    "classified_publications_bart_clean.parquet"
)
CACHE_PATH = (
    "/mnt/lab/unrestricted/muhammed.njie@marinemanagement.org.uk/published/"
    "govuk_summary_cache.parquet"
)

print("[INFO] Input parquet:", INPUT_PARQUET)
print("[INFO] Output parquet:", OUTPUT_PARQUET)
print("[INFO] Cache parquet :", CACHE_PATH)


# ============================================================
# 1. MMO Domain Lexicon (Short labels only)
# ============================================================
mmo_lexicon = {
    "Compliance": "compliance enforcement offences fines inspection illegal blue book catch record logbook sales notes",
    "FVL": "fishing vessel licence under10 over10 vessel registration permit capacity",
    "Marine Planning": "marine planning regional plans seascape plan area spp",
    "IIU": "illegal unreported unregulated iuu catch certificate processing statement storage document export",
    "SIA": "single issuance authority foreign vessel permit authorization eu waters",
    "Planning Team": "coastal concordat marine plan development workshops decision making",
    "Conservative team": "protected area mpa mcz wildlife byelaw conservation",
    "OMT": "operations coastal day to day management deployment monitoring",
    "IVMS": "ivms tracking monitoring system under12 gps device",
    "Corporate": "strategy privacy transparency annual report corporate risk",
    "Stats": "statistics datasets quality assessment landings data",
    "Comms": "communications newsletter press announcements media sea views",
    "Funding": "funding grants seafood fund emff mff financial support",
    "Global Marine Team": "blue belt overseas territory conservation international",
    "FMT": "fisheries management team quota landing obligation discard ban",
    "FDST": "data systems ers logbooks sales notes data collection",
    "Marine Licensing": "marine licence dredging construction disposal impact assessment"
}

# Short labels only (better BART confidence)
candidate_labels = list(mmo_lexicon.keys())
print("[INFO] Candidate labels loaded:", candidate_labels)


# ============================================================
# 2. GOV.UK SCRAPER
# ============================================================
def get_gov_summary(url, max_chars=2000):
    """
    Clean GOV.UK scraper:
    - Removes navigation, cookies, footer, related content
    - Extracts only meaningful guidance text
    """
    print(f"[SCRAPER] Fetching URL: {url}")
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        print("[SCRAPER] Page fetched successfully.")
    except Exception:
        print("[SCRAPER][WARN] Failed to fetch page → empty summary returned.")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    NOISE = [
        "nav", "header", "footer", "aside",
        ".gem-c-breadcrumbs", ".gem-c-related-navigation",
        ".app-c-related-navigation", ".gem-c-metadata",
        ".gem-c-publisher-metadata", ".gem-c-contextual-menu",
        ".gem-c-title", ".govuk-breadcrumbs", ".govuk-footer",
        ".govuk-header", ".gem-c-layout-super-navigation-header",
        ".gem-c-layout-super-navigation-footer", ".gem-c-topic-list",
        ".gem-c-document-list", ".gem-c-figure", "#related-content",
        "#full-page-navigation", ".support-links", ".govuk-accordion",
        ".app-c-contents-list", "#global-cookie-message"
    ]

    print("[SCRAPER] Removing boilerplate components...")
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

    main = soup.select_one(".govuk-main-wrapper, .govuk-grid-column-two-thirds")
    if not main:
        main = soup.body
    if not main:
        print("[SCRAPER][WARN] No main content found.")
        return ""

    # Strip <a> but keep text
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

    # Deduplicate
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
# 4. LOAD PARQUET AND BUILD Text_For_Model (with caching)
# ============================================================
print("[INFO] Loading MMO parquet...")
df = spark.read.parquet(INPUT_PARQUET)

pdf = df.toPandas()
print("[INFO] Parquet loaded. Total rows:", len(pdf))

if "URL" in pdf.columns:
    print("[INFO] Fetching GOV.UK summaries with cache...")
    pdf["Summary"] = pdf["URL"].apply(get_gov_summary_cached)
else:
    print("[WARN] URL column missing → Summary will be empty.")
    pdf["Summary"] = ""

pdf["Text_For_Model"] = (
    pdf.get("Title", pd.Series([""] * len(pdf))).fillna("").astype(str)
    + " "
    + pdf["Summary"].fillna("").astype(str)
).str.strip()

# Persist the updated cache (include a timestamp)
print("[CACHE] Persisting updated cache to Parquet...")
updated_cache_df = pd.DataFrame({
    "URL": list(cache.keys()),
    "Summary": list(cache.values()),
    "Last_Scraped": datetime.utcnow().isoformat()
})
# Drop duplicates just in case
updated_cache_df = updated_cache_df.drop_duplicates(subset=["URL"])
updated_cache_df.to_parquet(CACHE_PATH, index=False)
print("[CACHE] Cache saved:", CACHE_PATH)

# Back to Spark
spark_df = spark.createDataFrame(pdf)


# ============================================================
# 5. BART CLASSIFIER (Structured Outputs + Review Flags + Verbose Logs)
# ============================================================
MIN_CONF = 0.30          # relaxed confidence threshold
MARGIN_THRESHOLD = 0.05  # relaxed margin threshold

schema = StructType([
    StructField("bart_team", StringType(), True),
    StructField("bart_score", FloatType(), True),
    StructField("bart_margin", FloatType(), True),
    StructField("bart_final", StringType(), True),
    StructField("review_flag", StringType(), True)
])

_bart_singleton = None

@pandas_udf(schema)
def classify_bart_udf(texts: pd.Series) -> pd.DataFrame:
    """
    Classifies text into MMO teams using BART MNLI.
    Outputs:
        bart_team   - raw top prediction
        bart_score  - confidence of top label
        bart_margin - separation from 2nd best label
        bart_final  - accepted/unclassified after thresholds
        review_flag - automatic indicator for manual review
    """
    global _bart_singleton

    # Initialise BART once per Spark worker
    if _bart_singleton is None:
        print("\n[INFO] Initialising BART model for this Spark worker...")
        device = 0 if torch.cuda.is_available() else -1
        print(f"[INFO] Using device: {device} (0 = GPU, -1 = CPU)")
        _bart_singleton = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device
        )
        print("[INFO] BART model loaded successfully.")

    clf = _bart_singleton
    rows = []

    print(f"\n[INFO] Starting classification of {len(texts)} rows in this batch...")

    for idx, text in enumerate(texts):
        print(f"\n--- Processing row {idx+1} ---")

        # Handle empty text
        if not text or len(text.strip()) < 5:
            print("[WARN] Empty or too-short text → Unclassified + REVIEW_EMPTY")
            rows.append(("Unclassified", 0.0, 0.0, "Unclassified", "REVIEW_EMPTY"))
            continue

        # Hypothesis keeps it generic (labels are short)
        hypothesis_template = "This GOV.UK document relates to {}."

        # Call BART
        print("[INFO] Calling BART classifier...")
        out = clf(
            sequences=text,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
            truncation=True
        )

        labels = out["labels"]
        scores = out["scores"]

        # Extract top-1 and margin
        bart_team = labels[0]
        bart_score = float(scores[0])
        bart_margin = float(scores[0] - scores[1]) if len(scores) > 1 else bart_score

        print(f"[INFO] Top label: {bart_team}")
        print(f"[INFO] Score: {bart_score:.4f}, Margin: {bart_margin:.4f}")

        # Acceptance decision
        if bart_score >= MIN_CONF and bart_margin >= MARGIN_THRESHOLD:
            bart_final = bart_team
            print("[INFO] Decision: ACCEPTED")
        else:
            bart_final = "Unclassified"
            print("[INFO] Decision: REJECTED → Unclassified")

        # Review flags
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

        print(f"--- Row {idx+1} complete ---")

    print("[INFO] Batch classification complete. Returning results...")
    return pd.DataFrame(
        rows,
        columns=["bart_team", "bart_score", "bart_margin", "bart_final", "review_flag"]
    )


# ============================================================
# 6. EXECUTE CLASSIFIER + SAVE RESULTS
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
