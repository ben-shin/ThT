import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import re

# -----------------------------
# 1. Load CSV
# -----------------------------
# Replace 'ThT_DNAJB6.csv' with your file path
df = pd.read_csv("ThT_DNAJB6.csv")

# -----------------------------
# 2. Convert time column to minutes
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
# 3. Extract sample data
# -----------------------------
data = df.iloc[:, 1:].values.astype(float)  # all columns except Time

# -----------------------------
# 4. Define Boltzmann fit
# -----------------------------
def boltzmann(t, y0, ymax, k, t_half):
    return y0 + (ymax - y0) / (1 + np.exp(-k * (t - t_half)))

# -----------------------------
# 5. Process triplicates (every 3 columns = one set)
# -----------------------------
results = []

num_replicates = 3
for i in range(0, data.shape[1], num_replicates):
    trip = data[:, i:i+num_replicates]
    
    # Normalize each well individually
    F0 = trip[0, :]
    Fmax = trip.max(axis=0)
    norm = (trip - F0) / (Fmax - F0)
    
    # Average triplicates
    avg_curve = norm.mean(axis=1)
    
    # Smooth
    window_length = 11 if len(avg_curve) >= 11 else len(avg_curve)//2*2+1
    smoothed = savgol_filter(avg_curve, window_length, 2)
    
    # Fit Boltzmann
    p0 = [0, 1, 0.1, np.median(time)]
    params, _ = curve_fit(boltzmann, time, smoothed, p0=p0, maxfev=5000)
    
    results.append(params)
    
    # Optional plot
    plt.plot(time, avg_curve, 'o', alpha=0.3)
    plt.plot(time, smoothed, '-', label=f'Set {i//num_replicates + 1}')
    plt.plot(time, boltzmann(time, *params), '--')

plt.xlabel("Time (min)")
plt.ylabel("Normalized ThT fluorescence")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Compile results
# -----------------------------
results_df = pd.DataFrame(results, columns=['y0','ymax','k','t_half'])
print(results_df)
