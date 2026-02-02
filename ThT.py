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
df = pd.read_csv(args.file, header=None, skiprows=1)  # skip first header row

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
denom[denom == 0] = 1  # avoid divide by zero
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
window_length = 7  # must be odd
polyorder = 2
smoothed = savgol_filter(avg_windows, window_length=window_length, polyorder=polyorder, axis=0)

# -----------------------------
# 8. Remove first and last 3 time points
# -----------------------------
smoothed = smoothed[3:-3, :]
time_trimmed = time[3:-3]

# -----------------------------
# 9. Export to CSV
# -----------------------------
# Make sure number of sample names matches columns
if len(sample_names) != smoothed.shape[1]:
    print("Warning: number of sample names does not match number of columns after moving average!")
    if len(sample_names) > smoothed.shape[1]:
        sample_names = sample_names[:smoothed.shape[1]]
    else:
        sample_names += [f"Sample{i}" for i in range(smoothed.shape[1] - len(sample_names))]

output_df = pd.DataFrame(smoothed, columns=sample_names)
output_df.insert(0, "Time_min", time_trimmed)
output_file = "processed_tht.csv"
output_df.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")
