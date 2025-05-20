#needed pip install sdypy
#needed pip install pyFRF
#needed pip install nptdms
#needed pip install pandas

import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.signal import csd
from sdypy import EMA
import pyFRF
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
from tkinter import messagebox, simpledialog
import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)

# --- File Selection GUI ---
root = tk.Tk()
root.withdraw()  # Hide main window
file_path = filedialog.askopenfilename(
    title="Select a data file",
    filetypes=[
        ("Supported files", "*.tdms *.npy *.npz *.csv"),
        ("TDMS files", "*.tdms"),
        ("NumPy files", "*.npy *.npz"),
        ("CSV files", "*.csv")
    ]
)
root.destroy()
if not file_path:
    raise ValueError("No file selected.")

ext = os.path.splitext(file_path)[-1].lower()

# --- Load Data Based on File Extension ---
if ext == ".tdms":
    tdms = TdmsFile.read(file_path)
    t = tdms["TestGroup"]["Time"].data
    force = tdms["TestGroup"]["Force"].data
    acc = tdms["TestGroup"]["Acceleration"].data
    fs = 1 / (t[1] - t[0])

elif ext == ".csv":
    df = pd.read_csv(file_path)
    if not {'Time', 'Force', 'Acceleration'}.issubset(df.columns):
        raise ValueError("CSV must contain 'Time', 'Force', and 'Acceleration' columns.")
    t = df["Time"].values
    force = df["Force"].values
    acc = df["Acceleration"].values
    fs = 1 / (t[1] - t[0])

elif ext == ".npy":
    data = np.load(file_path, allow_pickle=True)
    t, acc = data
    raise NotImplementedError("'.npy' loading structure is unclear. Use '.npz' instead or modify this section.")

elif ext == ".npz":
    loaded = np.load(file_path)
    freqs = loaded["freqs"]
    H1_main = loaded["frf"]
    selected_response = 0
    frf = H1_main[0, selected_response, :]

    # Skip straight to EMA if data already FRF
    a = EMA.Model(frf, freqs, lower=10, upper=200, pol_order_high=60, frf_type='accelerance')
    a.get_poles()

    # --- Ask user for pole selection method ---
    temp_root = tk.Tk()
    temp_root.withdraw()
    use_gui = messagebox.askyesno(
        "Pole Selection",
        "Use stabilization chart to select poles?\n\nYes = Manual selection via chart\nNo = Enter frequencies"
    )
    temp_root.destroy()

    if use_gui:
        a.select_poles()
        plt.show(block=True)
    else:
        temp_root = tk.Tk()
        temp_root.withdraw()
        input_freqs = simpledialog.askstring(
            "Estimate Frequencies",
            "Enter estimated natural frequencies separated by commas (e.g., 40,100,130):",
            parent=temp_root
        )
        temp_root.destroy()

        if not input_freqs:
            raise ValueError("No frequencies entered.")
        natural_freqs_guess = [float(f.strip()) for f in input_freqs.split(",")]
        a.select_closest_poles(natural_freqs_guess)

    frf_rec, modal_const = a.get_constants(whose_poles='own', FRF_ind='all', upper_r=False)
    a.print_modal_data()

    freq_a = a.freq
    plt.figure(figsize=((10, 6)))
    plt.subplot(211)
    plt.semilogy(freqs, np.abs(frf.squeeze()), label='Experiment')
    plt.semilogy(freq_a, np.abs(frf_rec.squeeze()), '--', label='LSCF')
    plt.ylabel(r"abs($\alpha$)")
    plt.legend()
    plt.subplot(212)
    plt.plot(freqs, np.angle(frf.squeeze(), deg=1), label='Experiment')
    plt.plot(freq_a, np.angle(frf_rec.squeeze(), deg=1), '--', label='LSCF')
    plt.ylabel(r"angle($\alpha$)")
    plt.legend()
    plt.show()

    autoMAC = a.autoMAC()
    plt.matshow(np.abs(autoMAC), cmap="viridis", vmin=0.8, vmax=1.1)
    plt.colorbar(label='MAC Value')
    plt.title("autoMAC Matrix")
    plt.show()

    exit()

else:
    raise ValueError(f"Unsupported file type: {ext}")

# --- Plot Time-Domain Signals ---
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, force, label='Force', color='orange')
plt.ylabel("Force [a.u.]")
plt.title("Time-Domain Force Signal")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, acc, label='Acceleration', color='blue')
plt.ylabel("Acceleration [m/s²]")
plt.xlabel("Time [s]")
plt.title("Time-Domain Acceleration Signal")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Compute FRF if input is not one---
f, Pxy = csd(acc, force, fs=fs, nperseg=4096)
_, Pxx = csd(force, force, fs=fs, nperseg=4096)
H1 = Pxy / Pxx

# --- Continue with EMA ---
a = EMA.Model(H1, f, lower=10, upper=200, pol_order_high=60, frf_type='accelerance')
a.get_poles()

# --- Ask user for pole selection method ---
temp_root = tk.Tk()
temp_root.withdraw()
use_gui = messagebox.askyesno(
    "Pole Selection",
    "Use stabilization chart to select poles?\n\nYes = Manual selection via chart\nNo = Enter frequencies"
)
temp_root.destroy()

if use_gui:
    a.select_poles()
    plt.show(block=True)
else:
    temp_root = tk.Tk()
    temp_root.withdraw()
    input_freqs = simpledialog.askstring(
        "Estimate Frequencies",
        "Enter estimated natural frequencies separated by commas (e.g., 40,100,130):",
        parent=temp_root
    )
    temp_root.destroy()

    if not input_freqs:
        raise ValueError("No frequencies entered.")
    natural_freqs_guess = [float(f.strip()) for f in input_freqs.split(",")]
    a.select_closest_poles(natural_freqs_guess)

frf_rec, modal_const = a.get_constants(whose_poles='own', FRF_ind='all', upper_r=False)
a.print_modal_data()

freq_a = a.freq
plt.figure(figsize=((10, 6)))
plt.subplot(211)
plt.semilogy(f, np.abs(H1.squeeze()), label='Experiment')
plt.semilogy(freq_a, np.abs(frf_rec.squeeze()), '--', label='LSCF')
plt.ylabel(r"abs($\alpha$)")
plt.legend()

plt.subplot(212)
plt.plot(f, np.angle(H1.squeeze(), deg=1), label='Experiment')
plt.plot(freq_a, np.angle(frf_rec.squeeze(), deg=1), '--', label='LSCF')
plt.ylabel(r"angle($\alpha$)")
plt.legend()
plt.show()

autoMAC = a.autoMAC()
plt.matshow(np.abs(autoMAC), cmap="viridis")
plt.colorbar(label='MAC Value')
plt.show()

#Note: close to 1.0 means experimental modes and the selected/reconstructed FRF coincide
#MAC can also be used to compare between experiment and FEM
#Or it can just be used to check & reconstruct experimental result FRFw