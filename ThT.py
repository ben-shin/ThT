import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load your data
# -----------------------------
# Example: Excel file with time in column A and triplicates in columns B-D
# Replace 'your_file.xlsx' and sheet name
df = pd.read_excel('ThT_DNAJB6.xlsx', sheet_name='Sheet1')

time = df.iloc[:, 0].values          # Time column
triplicates = df.iloc[:, 1:4].values # Columns B-D

# -----------------------------
# 2. Normalize each triplicate
# -----------------------------
# Normalization: (F - F0) / (Fmax - F0)
F0 = triplicates[:, 0][:, np.newaxis]   # baseline per well
Fmax = triplicates.max(axis=0)          # max per well

norm = (triplicates - F0) / (Fmax - F0)

# -----------------------------
# 3. Average the triplicates
# -----------------------------
avg_curve = np.mean(norm, axis=1)
std_curve = np.std(norm, axis=1)

# -----------------------------
# 4. Optional: Savitzky-Golay smoothing
# -----------------------------
# window_length must be odd and <= len(time)
window_length = 11  # adjust depending on sampling
polyorder = 2
smoothed = savgol_filter(avg_curve, window_length, polyorder)

# -----------------------------
# 5. Sigmoidal (Boltzmann) function
# -----------------------------
def boltzmann(t, y0, ymax, k, t_half):
    return y0 + (ymax - y0) / (1 + np.exp(-k * (t - t_half)))

# Initial parameter guesses
p0 = [0, 1, 0.1, np.median(time)]

# Fit the smoothed curve
params, cov = curve_fit(boltzmann, time, smoothed, p0=p0)

y0_fit, ymax_fit, k_fit, t_half_fit = params

print(f"Baseline (y0): {y0_fit:.3f}")
print(f"Plateau (ymax): {ymax_fit:.3f}")
print(f"Rate constant (k): {k_fit:.3f}")
print(f"Half-time (t1/2): {t_half_fit:.2f} min")

# -----------------------------
# 6. Plot results
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(time, avg_curve, 'o', label='Averaged data', alpha=0.5)
plt.plot(time, smoothed, '-', label='Smoothed')
plt.plot(time, boltzmann(time, *params), '--', label='Boltzmann fit')
plt.fill_between(time, avg_curve - std_curve, avg_curve + std_curve, color='gray', alpha=0.2)
plt.xlabel('Time (min)')
plt.ylabel('Normalized ThT fluorescence')
plt.legend()
plt.tight_layout()
plt.show()
