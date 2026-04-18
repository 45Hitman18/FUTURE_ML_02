"""
notebooks/01_explore_data.py
EDA script for the Customer Support Ticket dataset.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "customer_support_tickets.csv")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

TEXT_COLUMN     = "Ticket Description"
CATEGORY_COLUMN = "Ticket Type"
PRIORITY_COLUMN = "Ticket Priority"

# 1. Load
print("[EDA] Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# 2. Basic info
print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nDtypes:\n", df.dtypes.to_string())
print("\nFirst 5 rows:\n", df.head().to_string())
missing = df.isnull().sum()
print("\nMissing values:\n", missing[missing > 0].to_string() if missing.sum() > 0 else "  None")

# 3. Identify columns
print(f"\nText column     : '{TEXT_COLUMN}'  (exists={TEXT_COLUMN in df.columns})")
print(f"Category column : '{CATEGORY_COLUMN}' (exists={CATEGORY_COLUMN in df.columns})")
print(f"Priority column : '{PRIORITY_COLUMN}'  (exists={PRIORITY_COLUMN in df.columns})")

# 4. Value counts
if CATEGORY_COLUMN in df.columns:
    print(f"\nCategory distribution:\n{df[CATEGORY_COLUMN].value_counts().to_string()}")
if PRIORITY_COLUMN in df.columns:
    print(f"\nPriority distribution:\n{df[PRIORITY_COLUMN].value_counts().to_string()}")

# 5. Charts
os.makedirs(DATA_DIR, exist_ok=True)

if CATEGORY_COLUMN in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    df[CATEGORY_COLUMN].value_counts().plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title("Ticket Category Distribution", fontsize=14)
    ax.set_xlabel("Category"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    p = os.path.join(DATA_DIR, "category_distribution.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n[EDA] Category chart saved: {p}")

if PRIORITY_COLUMN in df.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    df[PRIORITY_COLUMN].value_counts().plot(kind="bar", ax=ax,
        color=["#DD4949","#F5A623","#4CAF50"], edgecolor="white")
    ax.set_title("Ticket Priority Distribution", fontsize=14)
    ax.set_xlabel("Priority"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    p = os.path.join(DATA_DIR, "priority_distribution.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[EDA] Priority chart saved: {p}")

# 6. Summary
n_cat = df[CATEGORY_COLUMN].nunique() if CATEGORY_COLUMN in df.columns else "?"
n_pri = df[PRIORITY_COLUMN].nunique() if PRIORITY_COLUMN in df.columns else 3
print(f"\n[EDA] Dataset has {len(df):,} tickets across {n_cat} categories and {n_pri} priority levels.")
