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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    with TdmsFile.open(file_path) as tdms_file:
        #print(tdms_file.groups())  # List all groups
        #print(tdms_file["Group1"].channels())  # List all channels in Group1
        group1 = tdms_file["Group1"]
        t = group1["Time"][:]

        # List of response channel names
        response_channels = [
        "L6-Z",
        ]
        acc_list = []

        #acc_forEMA = group1["L5-Z"][:]
        for ch_name in response_channels:
            if ch_name in group1:
                acc_list.append(group1[ch_name][:])
            else:
                raise KeyError(f"Channel {ch_name} not found in TDMS file.")

    # Compute sampling frequency
    fs = 1 / (t[1] - t[0])          #512 Hz for GVT

    '''
    #single sensor
    f_forEMA, Pxy_forEMA = csd(acc_forEMA, force, fs=fs, nperseg=nperseg)
    _, Pxx_forEMA = csd(force, force, fs=fs, nperseg=nperseg)
    H1_forEMA = Pxy_forEMA / Pxx_forEMA
    '''

elif ext == ".npy":
    data = np.load(file_path, allow_pickle=True)
    t, acc = data
    raise NotImplementedError("'.npy' loading structure is unclear. Use '.npz' instead or modify this section.")

elif ext == ".npz":
    loaded = np.load(file_path)
    freqs = loaded["freqs"]
    H1_array = loaded["frf"]
    #Probably need change for multiple sensors
    f = freqs
    exit()

else:
    raise ValueError(f"Unsupported file type: {ext}")


# --- Sensor Coordinates (in mm) ---
print(f"Number of acceleration channels loaded: {len(acc_list)}")
for i, acc in enumerate(acc_list):
    print(f"Acc {i} shape: {np.shape(acc)}")
    print(acc)

print(f"Time vector shape: {t.shape}")

# --- Plot Time-Domain Acceleration Signals from experiment ---
plt.figure(figsize=(12, 6))

for i, acc in enumerate(acc_list):
    if len(acc) == len(t):  # Only plot if lengths match
        plt.plot(t, acc, label=f'Acc L{i+3}-Z')
    else:
        print(f"Skipping Acc L{i+3}-Z: length mismatch (acc={len(acc)}, t={len(t)})")

plt.ylabel("Acceleration [m/s²]")
plt.xlabel("Time [s]")
plt.title("Time-Domain Acceleration Signals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


