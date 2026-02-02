#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import argparse
import re

# -----------------------------
# 1. Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Process ThT CSV data with triplicates")
parser.add_argument("--file", required=True, help="Path to CSV file")
parser.add_argument("--samples", required=True, help="Comma-separated sample names (one per triplicate group)")
args = parser.parse_args()

sample_names = args.samples.split(",")

# -----------------------------
# 2. Load CSV and remove header row
# -----------------------------
df = pd.read_csv(args.file, header=None, skiprows=1)  # skip the first header row

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
# 6. Average fixed triplicate groups
# -----------------------------
num_replicates = 3
num_groups = norm.shape[1] // num_replicates  # number of triplicate groups
avg_groups = np.zeros((norm.shape[0], num_groups))

for i in range(num_groups):
    start = i * num_replicates
    end = start + num_replicates
    avg_groups[:, i] = norm[:, start:end].mean(axis=1)

# -----------------------------
# 7. Savitzky-Golay filter
# -----------------------------
window_length = 7  # must be odd and smaller than number of rows
if window_length > avg_groups.shape[0]:
    window_length = avg_groups.shape[0] // 2 * 2 + 1  # largest odd number less than num rows

polyorder = 2
smoothed = savgol_filter(avg_groups, window_length=window_length, polyorder=polyorder, axis=0)

# -----------------------------
# 8. Remove first and last 3 time points
# -----------------------------
trim = 3
if smoothed.shape[0] <= 2*trim:
    print("Warning: Not enough points to trim for Savitzky-Golay filter. Skipping trim.")
    smoothed_trimmed = smoothed
    time_trimmed = time
else:
    smoothed_trimmed = smoothed[trim:-trim, :]
    time_trimmed = time[trim:-trim]

# -----------------------------
# 9. Export to CSV
# -----------------------------
# Make sure number of sample names matches columns
if len(sample_names) != smoothed_trimmed.shape[1]:
    print("Warning: number of sample names does not match number of triplicate groups!")
    if len(sample_names) > smoothed_trimmed.shape[1]:
        sample_names = sample_names[:smoothed_trimmed.shape[1]]
    else:
        sample_names += ["Sample{}".format(i) for i in range(smoothed_trimmed.shape[1] - len(sample_names))]

output_df = pd.DataFrame(smoothed_trimmed, columns=sample_names)
output_df.insert(0, "Time_min", time_trimmed)
output_file = "processed_tht.csv"
output_df.to_csv(output_file, index=False)
print(f"Processed data saved to {}".format(output_file))
