#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import argparse

# -----------------------------
# 1. Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Fit ThT processed CSV to sigmoidal curves")
parser.add_argument("--file", required=True, help="Path to processed CSV file")
parser.add_argument("--samples", required=True, help="Comma-separated sample names")
args = parser.parse_args()

sample_names = args.samples.split(",")

# -----------------------------
# 2. Load CSV
# -----------------------------
df = pd.read_csv(args.file)

# -----------------------------
# 3. Define sigmoidal function
# -----------------------------
def sigmoid(x, A, B, x0, k):
    """
    A: min
    B: max - min
    x0: midpoint
    k: slope
    """
    return A + B / (1 + np.exp(-(x - x0) / k))

# -----------------------------
# 4. Prepare x values (time)
# -----------------------------
x = df.iloc[:, 0].values  # first column is Time_min

# -----------------------------
# 5. Fit each column
# -----------------------------
results = []
skipped_samples = []

for col_name in sample_names:
    if col_name not in df.columns:
        print(f"Column {col_name} not found in CSV. Skipping.")
        skipped_samples.append(col_name)
        continue

    y = df[col_name].values

    # Skip if all zeros or constant array
    if np.all(y == 0) or np.all(y == y[0]):
        print(f"Skipping {col_name}: no signal or constant array")
        skipped_samples.append(col_name)
        continue

    # Initial guess: min, max-min, midpoint, slope
    A0 = np.min(y)
    B0 = np.max(y) - A0
    x0 = x[np.argmax(np.diff(y))] if np.any(np.diff(y)) else np.median(x)
    k0 = 1.0
    p0 = [A0, B0, x0, k0]

    try:
        params, cov = curve_fit(sigmoid, x, y, p0=p0, maxfev=10000)
        perr = np.sqrt(np.diag(cov))  # standard errors
        results.append({
            "Sample": col_name,
            "A": params[0], "A_err": perr[0],
            "B": params[1], "B_err": perr[1],
            "x0": params[2], "x0_err": perr[2],
            "k": params[3], "k_err": perr[3]
        })
        print(f"Fit successful for {col_name}")
    except Exception as e:
        print(f"Fit failed for {col_name}: {e}")
        skipped_samples.append(col_name)

# -----------------------------
# 6. Save results
# -----------------------------
if results:
    out_df = pd.DataFrame(results)
    out_file = "tht_fit_results.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Fitting results saved to {out_file}")

if skipped_samples:
    print(f"Skipped samples ({len(skipped_samples)}): {', '.join(skipped_samples)}")
