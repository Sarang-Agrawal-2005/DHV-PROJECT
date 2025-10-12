import os
import joblib
import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# JOBLIB PATCH (Windows safe, disable multiprocessing issues)
# ------------------------------------------------------------------
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["JOBLIB_N_JOBS"] = "1"

_real_init = joblib.Parallel.__init__
def _safe_init(self, *args, **kwargs):
    # Force sequential execution
    kwargs["n_jobs"] = 1
    return _real_init(self, *args, **kwargs)
joblib.Parallel.__init__ = _safe_init

# ------------------------------------------------------------------
# LOAD REAL DATA
# ------------------------------------------------------------------
df = pd.read_csv("heart.csv")

# ------------------------------------------------------------------
# STEP 1. DEFINE METADATA
# ------------------------------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Target
metadata.update_column("HeartDisease", sdtype="categorical")

# Categorical columns (adjust if your dataset differs)
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
for col in categorical_cols:
    if col in df.columns:
        metadata.update_column(col, sdtype="categorical")

# Numeric columns
numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
for col in numeric_cols:
    if col in df.columns:
        metadata.update_column(col, sdtype="numerical", computer_representation="Float")

# ------------------------------------------------------------------
# STEP 2. TRAIN SYNTHESIZER
# ------------------------------------------------------------------
synth = TVAESynthesizer(
    metadata,
    epochs=900,
    enforce_min_max_values=True,
    batch_size=256,
    embedding_dim=256,
    compress_dims=(256, 128),
    decompress_dims=(128, 256),
)

print("🚀 Training TVAE...")
synth.fit(df)
print("✅ TVAE training complete.")

# ------------------------------------------------------------------
# STEP 3. UNCONDITIONAL SAMPLING + BALANCING
# ------------------------------------------------------------------
n = 250000  # oversample then filter
print(f"Sampling {n} synthetic rows (unconditional)...")
synthetic = synth.sample(n)

# Ensure target is binary int
synthetic["HeartDisease"] = synthetic["HeartDisease"].round().astype(int)

# Match real distribution
counts_real = df["HeartDisease"].value_counts(normalize=True)
n0 = int(n * counts_real.get(0, 0))
n1 = int(n * counts_real.get(1, 0))

synthetic0 = synthetic[synthetic["HeartDisease"] == 0].sample(n=n0, replace=True, random_state=42)
synthetic1 = synthetic[synthetic["HeartDisease"] == 1].sample(n=n1, replace=True, random_state=42)
synthetic = pd.concat([synthetic0, synthetic1], ignore_index=True)

print(f"Balanced synthetic dataset: {synthetic['HeartDisease'].value_counts().to_dict()}")

# ------------------------------------------------------------------
# STEP 4. POST-SAMPLE QUALITY FILTER
# ------------------------------------------------------------------
print("🔎 Filtering synthetic data for higher fidelity...")

mix = pd.concat([df.assign(_y=1), synthetic.assign(_y=0)], ignore_index=True)
X = pd.get_dummies(mix.drop(columns=["_y"]), drop_first=True)
y = mix["_y"].values

X_tr, X_va, y_tr, y_va = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_tr, y_tr)

# Predict "synthetic probability"
proba_synth_is_synth = clf.predict_proba(X.loc[synthetic.index])[:, 1]

# Keep rows not too obviously fake
keep = proba_synth_is_synth < 0.8
synthetic_filtered = synthetic.loc[keep].copy()

print(f"💡 Filtered synthetic rows: {len(synthetic_filtered)} / {len(synthetic)}")

# ------------------------------------------------------------------
# STEP 5. SAVE
# ------------------------------------------------------------------
synthetic_filtered.to_csv("synthetic.csv", index=False)
print("✅ Saved high-quality synthetic_tvae.csv")
