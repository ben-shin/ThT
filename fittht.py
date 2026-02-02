#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import argparse

# -----------------------------
# Sigmoid function (Boltzmann)
# -----------------------------
def boltzmann(t, Fmin, Fmax, t_half, k):
    return Fmin + (Fmax - Fmin) / (1 + np.exp(-(t - t_half)/k))

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Fit ThT CSV data to sigmoidal curves")
parser.add_argument("--file", required=True, help="Path to CSV file")
parser.add_argument("--samples", required=True, help="Comma-separated sample names")
args = parser.parse_args()

sample_names = args.samples.split(",")

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(args.file)
time = df.iloc[:, 0].values  # first column is time
data = df.iloc[:, 1:].values  # the sample columns

# -----------------------------
# Prepare results storage
# -----------------------------
results = []

# -----------------------------
# Fit each column
# -----------------------------
for i, col_name in enumerate(df.columns[1:]):
    y = data[:, i]

    # Skip columns with no variation
    if np.all(np.isnan(y)) or np.ptp(y) < 1e-8:
        print(f"Skipping column {col_name}: no variation")
        continue

    # Initial guess for parameters
    Fmin_guess = np.min(y)
    Fmax_guess = np.max(y)
    t_half_guess = time[np.argmax(y >= (Fmax_guess + Fmin_guess)/2)]
    k_guess = 1.0
    p0 = [Fmin_guess, Fmax_guess, t_half_guess, k_guess]

    try:
        params, cov = curve_fit(boltzmann, time, y, p0=p0, maxfev=5000)
        perr = np.sqrt(np.diag(cov))  # standard errors
        results.append({
            "Sample": col_name,
            "Fmin": params[0], "Fmin_SE": perr[0],
            "Fmax": params[1], "Fmax_SE": perr[1],
            "t_half": params[2], "t_half_SE": perr[2],
            "k": params[3], "k_SE": perr[3],
        })
        print(f"Fitted {col_name}")
    except Exception as e:
        print(f"Fit failed for {col_name}: {e}")

# -----------------------------
# Save results
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("tht_fit_results.csv", index=False)
print("Fitting results saved to tht_fit_results.csv")
