#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import argparse

# -----------------------------
# 1. Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Fit ThT data to sigmoidal curve")
parser.add_argument("--file", required=True, help="Path to processed CSV file")
parser.add_argument("--samples", required=True, help="Comma-separated sample names")
args = parser.parse_args()

sample_names = args.samples.split(",")

# -----------------------------
# 2. Load CSV
# -----------------------------
df = pd.read_csv(args.file)

# -----------------------------
# 3. Drop rows with NaNs in Time_min or sample columns
# -----------------------------
all_cols_to_check = ["Time_min"] + sample_names
df_clean = df.dropna(subset=all_cols_to_check)
dropped_rows = len(df) - len(df_clean)
if dropped_rows > 0:
    print(f"Dropped {dropped_rows} rows containing NaNs in Time_min or sample columns.")

# -----------------------------
# 4. Define sigmoidal function (Boltzmann)
# -----------------------------
def boltzmann(x, A1, A2, x0, dx):
    return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

# -----------------------------
# 5. Fit each column
# -----------------------------
results = []
time = df_clean["Time_min"].values

for col_name in sample_names:
    y = df_clean[col_name].values

    # Skip entirely empty columns
    if np.all(np.isnan(y)):
        print(f"Skipping {col_name}: all values are NaN")
        continue

    # Initial guesses
    p0 = [y.max(), y.min(), time[len(time)//2], 1.0]

    try:
        params, cov = curve_fit(boltzmann, time, y, p0=p0, maxfev=5000)
        errors = np.sqrt(np.diag(cov))
        results.append({
            "Sample": col_name,
            "A1": params[0], "A1_err": errors[0],
            "A2": params[1], "A2_err": errors[1],
            "x0": params[2], "x0_err": errors[2],
            "dx": params[3], "dx_err": errors[3]
        })
    except Exception as e:
        print(f"Fit failed for {col_name}: {e}")

# -----------------------------
# 6. Export fitting results
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("tht_fit_results.csv", index=False)
print("Fitting results saved to tht_fit_results.csv")
