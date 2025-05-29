import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

def load_frf_file(prompt_title):
    # GUI file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=prompt_title,
        filetypes=[("FRF files", "*.npz *.csv")]
    )
    root.destroy()
    if not file_path:
        raise ValueError("No file selected.")

    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".npz":
        data = np.load(file_path)
        freqs = data["freqs"]
        frf = data["frf"].squeeze()
    elif ext == ".csv":
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 3:
            raise ValueError("CSV must have at least 3 columns: frequency, real, imag")
        freqs = df.iloc[:, 0].values
        real = df.iloc[:, 1].values
        imag = df.iloc[:, 2].values
        frf = real + 1j * imag
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return freqs, frf, os.path.basename(file_path)

# --- Load experimental FRF ---
freq_exp, frf_exp, label_exp = load_frf_file("Select Experimental FRF")
print(f"[{label_exp}] Experimental FRF shape: freqs = {freq_exp.shape}, frf = {frf_exp.shape}")

# --- Load FEM FRF ---
freq_fem, frf_fem, label_fem = load_frf_file("Select FEM FRF")
print(f"[{label_fem}] FEM FRF shape: freqs = {freq_fem.shape}, frf = {frf_fem.shape}")

# --- Select one DOF (e.g., 0) ---
dof_index = 0
if frf_exp.shape[0] <= dof_index or frf_fem.shape[0] <= dof_index:
    raise IndexError("DOF index out of range for FRF data.")

# Select first DOF if 2D, or use as-is if 1D
frf_exp_dof = frf_exp[dof_index] if frf_exp.ndim == 2 else frf_exp
frf_fem_dof = frf_fem[dof_index] if frf_fem.ndim == 2 else frf_fem

# --- Normalize by max magnitude ---
frf_exp_mag = np.abs(frf_exp_dof)
frf_fem_mag = np.abs(frf_fem_dof)

frf_exp_norm = frf_exp_mag / np.max(frf_exp_mag)
frf_fem_norm = frf_fem_mag / np.max(frf_fem_mag)

# --- Plot ---
plt.figure(figsize=(10, 6))

# Magnitude plot (normalized)
plt.subplot(2, 1, 1)
plt.semilogy(freq_exp, frf_exp_norm, label=label_exp + " [DOF 0]")
plt.semilogy(freq_fem, frf_fem_norm, '--', label=label_fem + " [DOF 0]")
plt.ylabel("Normalized Magnitude")
plt.title("Normalized FRF Magnitude Comparison")
plt.legend()
plt.grid(True)

# Phase plot (unchanged)
plt.subplot(2, 1, 2)
plt.plot(freq_exp, np.angle(frf_exp_dof, deg=True), label=label_exp + " [DOF 0]")
plt.plot(freq_fem, np.angle(frf_fem_dof, deg=True), '--', label=label_fem + " [DOF 0]")
plt.ylabel("Phase [deg]")
plt.xlabel("Frequency [Hz]")
plt.title("FRF Phase Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()