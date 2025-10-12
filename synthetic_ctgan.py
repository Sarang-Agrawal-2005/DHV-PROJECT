import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["JOBLIB_N_JOBS"] = "1"

import pandas as pd
import pickle
from ctgan import CTGAN

# 1. Load dataset
df = pd.read_csv("heart_preprocessed.csv", low_memory=False)
print("✅ Dataset loaded with shape:", df.shape)

# 2. Detect categorical columns
categorical_columns = []
for col in df.columns:
    if df[col].dtype == "object":
        categorical_columns.append(col)
    else:
        if df[col].nunique() < 0.05 * len(df):
            categorical_columns.append(col)

# 3. Clean data
for col in df.columns:
    if col in categorical_columns:
        df[col] = df[col].astype(str).fillna("Unknown")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

# 4. Train CTGAN (single-threaded)
model = CTGAN(epochs=500)
model.fit(df, discrete_columns=categorical_columns)
print("✅ CTGAN model trained with categorical columns:", categorical_columns)

# 5. Generate synthetic samples
synthetic_data = model.sample(1000)
synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
print("✅ Synthetic data generated with 1000 samples")

# 6. Save synthetic dataset
synthetic_df.to_csv("synthetic_data.csv", index=False)

# 7. Save CTGAN model
with open("ctgan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Synthetic dataset saved as synthetic_data.csv")
print("✅ Model saved as ctgan_model.pkl")
print("🔹 Categorical columns detected:", categorical_columns)
