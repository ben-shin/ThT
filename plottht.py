#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# -----------------------------
# 1. Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Plot ThT data with sigmoidal fits")
parser.add_argument("--file", required=True, help="Path to processed CSV file")
parser.add_argument("--samples", required=True, help="Comma-separated sample names")
args = parser.parse_args()

sample_names = args.samples.split(",")

# -----------------------------
# 2. Load processed ThT data
# -----------------------------
df = pd.read_csv(args.file)
time = df["Time_min"].values

# -----------------------------
# 3. Load fitting results
# -----------------------------
fit_df = pd.read_csv("tht_fit_results.csv")

# -----------------------------
# 4. Define Boltzmann function
# -----------------------------
def boltzmann(x, A1, A2, x0, dx):
    return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

# -----------------------------
# 5. Plot each sample
# -----------------------------
plt.figure(figsize=(8,6))

for sample in sample_names:
    if sample not in df.columns:
        print(f"Warning: {sample} not found in data. Skipping.")
        continue

    y = df[sample].values

    # Plot data points
    plt.scatter(time, y, label=f"{sample} data", s=30)

    # Plot fitted curve if available
    fit_row = fit_df[fit_df["Sample"] == sample]
    if not fit_row.empty:
        A1 = fit_row["A1"].values[0]
        A2 = fit_row["A2"].values[0]
        x0 = fit_row["x0"].values[0]
        dx = fit_row["dx"].values[0]

        x_fit = np.linspace(time.min(), time.max(), 200)
        y_fit = boltzmann(x_fit, A1, A2, x0, dx)
        plt.plot(x_fit, y_fit, label=f"{sample} fit")

plt.xlabel("Time (min)")
plt.ylabel("Normalized fluorescence")
plt.title("ThT aggregation kinetics")
plt.legend()
plt.tight_layout()
plt.show()
