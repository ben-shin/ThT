#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import argparse
import re

# -----------------------------
# 1. Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Process ThT CSV data")
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
# 4. Extract numeric data
# -----------------------------
data = df.iloc[:, 1:].astype(float).values

# -----------------------------
# 5. Normalize each column
# -----------------------------
F0 = data[0, :]
Fmax = np.nanmax(data, axis=0)
denom = Fmax - F0
denom[denom <= 1e-8] = 1.0  # avoid divide by zero
norm = (data - F0) / denom
norm = np.clip(norm, 0, 1)  # ensure all normalized values in [0,1]

# -----------------------------
# 6. Moving window average over triplicates
# -----------------------------
triplicates = 3
num_samples = norm.shape[1] // triplicates
avg_windows = np.zeros((norm.shape[0], num_samples))

for i in range(num_samples):
    start = i * triplicates
    end = start + triplicates
    avg_windows[:, i] = np.nanmean(norm[:, start:end], axis=1)  # safe average

# -----------------------------
# 7. Savitzky-Golay filter
# -----------------------------
window_length = 7  # must be odd
polyorder = 2
# Apply SG filter column-wise
smoothed = np.zeros_like(avg_windows)
for i in range(avg_windows.shape[1]):
    col = avg_windows[:, i]
    # Only apply if enough points
    if len(col) >= window_length:
        smoothed[:, i] = savgol_filter(col, window_length=window_length, polyorder=polyorder)
    else:
        smoothed[:, i] = col

# -----------------------------
# 8. Remove first and last 3 time points
# -----------------------------
smoothed_trimmed = smoothed[3:-3, :]
time_trimmed = time[3:-3]

# -----------------------------
# 9. Fix NaNs and negatives
# -----------------------------
smoothed_trimmed = np.nan_to_num(smoothed_trimmed, nan=0.0, posinf=0.0, neginf=0.0)
smoothed_trimmed[smoothed_trimmed < 0] = 0.0
smoothed_trimmed = np.clip(smoothed_trimmed, 0, 1)

# -----------------------------
# 10. Match sample names
# -----------------------------
if len(sample_names) != smoothed_trimmed.shape[1]:
    print("Warning: number of sample names does not match number of triplicate-averaged columns!")
    if len(sample_names) > smoothed_trimmed.shape[1]:
        sample_names = sample_names[:smoothed_trimmed.shape[1]]
    else:
        sample_names += [f"Sample{i}" for i in range(smoothed_trimmed.shape[1] - len(sample_names))]

# -----------------------------
# 11. Export CSV
# -----------------------------
output_df = pd.DataFrame(smoothed_trimmed, columns=sample_names)
output_df.insert(0, "Time_min", time_trimmed)
output_file = "processed_tht.csv"
output_df.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")
