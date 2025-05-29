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
    with TdmsFile.open(file_path) as tdms_file:
        #print(tdms_file.groups())  # List all groups
        #print(tdms_file["Group1"].channels())  # List all channels in Group1
        group1 = tdms_file["Group1"]
        t = group1["Time"][:]
        force = group1["ILF-Z"][:]

        # List of response channel names
        response_channels = [
        "L3-Z", "L4-Z", "L5-Z", "L6-Z", "L7-Z", "L8-Z", "L9-Z", "L10-Z", "L11-Z", "L12-Z",
        "L13-Z", "L14-Z", "L15-Z", "L16-Z", "L17-Z", "L18-Z", "L20-Z",
        "R3-Z", "R4-Z", "R5-Z", "R6-Z", "R7-Z", "R8-Z", "R9-Z", "R10-Z", "R11-Z", "R12-Z",
        "R13-Z", "R14-Z", "R15-Z", "R16-Z", "R17-Z", "R18-Z", "R20-Z"
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

    # Compute FRF for each sensor (output)
    H1_list = []
    n_points = len(force)
    nperseg = min(8192, n_points // 2)  # use a safe value
    for acc in acc_list:
        f, Pxy = csd(acc, force, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        _, Pxx = csd(force, force, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        H1 = Pxy / Pxx
        H1_list.append(H1)

    H1_array = np.stack(H1_list, axis=0)  # Shape: (outputs, freqs)
    #H1_array = H1_array[:, :, None]  # Shape: (outputs, freqs, 1 input)

    '''
    #single sensor
    f_forEMA, Pxy_forEMA = csd(acc_forEMA, force, fs=fs, nperseg=nperseg)
    _, Pxx_forEMA = csd(force, force, fs=fs, nperseg=nperseg)
    H1_forEMA = Pxy_forEMA / Pxx_forEMA
    '''

elif ext == ".csv":
    df = pd.read_csv(file_path)
    required_cols = {"Time", "Force", "L3-Z", "L4-Z", "L5-Z", "L6-Z"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    t = df["Time"].values
    force = df["Force"].values
    fs = 1 / (t[1] - t[0])

    acc_list = [df[ch].values for ch in ["L3-Z", "L4-Z", "L5-Z", "L6-Z"]]

    H1_list = []
    n_points = len(force)
    nperseg = min(4096, n_points // 2)  # use a safe value
    for acc in acc_list:
        f, Pxy = csd(acc, force, fs=fs, nperseg=nperseg)
        _, Pxx = csd(force, force, fs=fs, nperseg=nperseg)
        H1 = Pxy / Pxx
        H1_list.append(H1)

    H1_array = np.stack(H1_list, axis=0)
    H1_array = H1_array[:, :, None]

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

'''
# --- Plot Time-Domain Signals from experiment---
plt.ion()
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, force, label='Force', color='orange')
plt.ylabel("Force [N]")
plt.title("Time-Domain Force Signal")
plt.grid(True)

plt.subplot(2, 1, 2)
for i, acc in enumerate(acc_list):
    plt.plot(t, acc, label=f'Acc L{i+3}-Z')  # You can adjust labels if needed
plt.ylabel("Acceleration [m/s²]")
plt.xlabel("Time [s]")
plt.title("Time-Domain Acceleration Signals")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#Plot FFT

def plot_fft(signal, fs, title="FFT"):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.fft.rfft(signal)
    magnitude = np.abs(fft_vals)
    magnitude = np.abs(fft_vals) / n

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitude)
    plt.title(f"{title}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example: Plot FFT of force and acceleration
plot_fft(force, fs, title="FFT of Force")
for i, acc in enumerate(acc_list):
    plot_fft(acc, fs, title=f"FFT of Acceleration L{i+3}-Z")
'''

# --- Continue with EMA ---
a = EMA.Model(H1_array, f, lower=5, upper=80, pol_order_high=30, frf_type='accelerance')
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

#Reconstruct FRFs
frf_rec, modal_const = a.get_constants(whose_poles='own', FRF_ind='all', upper_r=False)
a.print_modal_data()

freq_a = a.freq


# Apply frequency cutoff
f_cutoff = 80
mask_exp = f <= f_cutoff
mask_rec = freq_a <= f_cutoff

plt.figure(figsize=(10, 6))

# Magnitude plot
plt.subplot(211)
plt.semilogy(f[mask_exp], np.abs(H1_array[0].squeeze())[mask_exp], label='Experiment')
plt.semilogy(freq_a[mask_rec], np.abs(frf_rec[0].squeeze())[mask_rec], '--', label='LSCF')
plt.ylabel(r"abs($\alpha$)")
plt.legend()
plt.title("FRF Magnitude up to 80 Hz")
plt.grid(True)

# Phase plot
plt.subplot(212)
plt.plot(f[mask_exp], np.angle(H1_array[0].squeeze(), deg=True)[mask_exp], label='Experiment')
plt.plot(freq_a[mask_rec], np.angle(frf_rec[0].squeeze(), deg=True)[mask_rec], '--', label='LSCF')
plt.ylabel(r"angle($\alpha$)")
plt.xlabel("Frequency [Hz]")
plt.legend()
plt.grid(True)
plt.title("FRF Phase up to 80 Hz")

plt.tight_layout()
plt.show()

autoMAC = a.autoMAC()
plt.matshow(np.abs(autoMAC), cmap="viridis")
plt.colorbar(label='MAC Value')
plt.show()

#Note: close to 1.0 means experimental modes and the selected/reconstructed FRF coincide
#In this case, it is only checking for mode duplicates/orthogonality
#But MAC can also be used to compare between experiment and FEM
#Or it can just be used to check & reconstruct experimental result FRFw

mode_shapes = a.mode_shapes  # Shape: (num_dofs, num_modes)

# Transpose for easier access: (modes, DOFs)
modes = mode_shapes.T

plt.figure(figsize=(12, 6))
for i, mode in enumerate(modes):
    plt.plot(np.abs(mode), label=f'Mode {i+1}')
plt.xlabel("Sensor Index (DOFs)")
plt.ylabel("Mode Shape Magnitude")
plt.title("Mode Shapes (Absolute Values)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Save Reconstructed FRF and Frequency Vector for Later Use ---
save_dir = r"C:\Nin Folder\RVC\IRIS-T\GVT Processing\src\VT_processing\Reconstructed_FRF"
base_name = os.path.splitext(os.path.basename(file_path))[0]
save_path = os.path.join(save_dir, f"{base_name}_reconstructed_frf.npz")

np.savez_compressed(
    save_path,
    freqs=freq_a,
    frf=frf_rec
)

print(f"\nReconstructed FRF and frequencies saved to:\n{save_path}")








