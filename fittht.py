#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import argparse
import re
import matplotlib.pyplot as plt

# -----------------------------
# 1. Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Process and fit ThT CSV data")
parser.add_argument("--file", required=True, help="Path to CSV file")
parser.add_argument("--samples", required=True, help="Comma-separated sample names")
args = parser.parse_args()

sample_names = args.samples.split(",")

# -----------------------------
# 2. Load CSV and remove header row
# -----------------------------
df = pd.read_csv(args.file, header=None, skiprows=1)

# -----------------------------
# 3. Convert first column to minutes
# -----------------------------
def time_to_min(s):
    match = re.match(r"(\d+)\s*h\s*(\d*)\s*min?", str(s))
    if match:
        h = int(match.group(1))
        m = int(match.group(2)) if match.group(2) else 0
        return h*60 + m
    return np.nan

time = df.iloc[:, 0].apply(time_to_min).values

# -----------------------------
# 4. Extract numeric sample data
# -----------------------------
data = df.iloc[:, 1:].values.astype(float)

# -----------------------------
# 5. Normalize each column
# -----------------------------
F0 = data[0, :]
Fmax = data.max(axis=0)
denom = Fmax - F0
denom[denom == 0] = 1
norm = (data - F0) / denom

# -----------------------------
# 6. Moving window average (3 columns)
# -----------------------------
num_replicates = 3
num_windows = norm.shape[1] - num_replicates + 1
avg_windows = np.zeros((norm.shape[0], num_windows))

for i in range(num_windows):
    avg_windows[:, i] = norm[:, i:i+num_replicates].mean(axis=1)

# -----------------------------
# 7. Savitzky-Golay filter
# -----------------------------
window_length = 7
polyorder = 2
smoothed = savgol_filter(avg_windows, window_length=window_length, polyorder=polyorder, axis=0)

# -----------------------------
# 8. Remove first and last 3 time points
# -----------------------------
smoothed = smoothed[3:-3, :]
time_trimmed = time[3:-3]

# -----------------------------
# 9. Fit Boltzmann sigmoid
# -----------------------------
def boltzmann(t, y0, ymax, k, t_half):
    return y0 + (ymax - y0) / (1 + np.exp(-k * (t - t_half)))

fit_results = []

for i in range(smoothed.shape[1]):
    y = smoothed[:, i]
    p0 = [0, 1, 0.1, np.median(time_trimmed)]
    try:
        params, cov = curve_fit(boltzmann, time_trimmed, y, p0=p0, maxfev=5000)
        errors = np.sqrt(np.diag(cov))
    except Exception as e:
        print(f"Fit failed for column {i}: {e}")
        params = [np.nan]*4
        errors = [np.nan]*4
    fit_results.append(params + list(errors))
    
    # Optional: plot each fit
    plt.plot(time_trimmed, y, 'o', alpha=0.3)
    plt.plot(time_trimmed, boltzmann(time_trimmed, *params), '-', label=sample_names[i] if i<len(sample_names) else f"Sample{i}")

plt.xlabel("Time (min)")
plt.ylabel("Normalized ThT fluorescence")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Export fit parameters
# -----------------------------
columns = ['y0','ymax','k','t_half','err_y0','err_ymax','err_k','err_t_half']
fit_df = pd.DataFrame(fit_results, columns=columns)

# Add sample names
fit_df.insert(0, "Sample", [sample_names[i] if i<len(sample_names) else f"Sample{i}" for i in range(smoothed.shape[1])])

fit_df.to_csv("tht_fit_results.csv", index=False)
print("Fitting results saved to tht_fit_results.csv")
