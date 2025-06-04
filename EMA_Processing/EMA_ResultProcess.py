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
        force = group1["ILF-Z"][:]

        # List of response channel names
        response_channels = [
        "L3-Z", "L4-Z", "L5-Z", "L6-Z", "L7-Z", "L8-Z", "L9-Z", "L10-Z", "L11-Z", "L12-Z",
        "L13-Z", "L14-Z", "L15-Z", "L16-Z", "L17-Z", "L18-Z",
        "R3-Z", "R4-Z", "R5-Z", "R6-Z", "R7-Z", "R8-Z", "R9-Z", "R10-Z", "R11-Z", "R12-Z",
        "R13-Z", "R14-Z", "R15-Z", "R16-Z", "R17-Z", "R18-Z",
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


# --- Sensor Coordinates (in mm) ---
mm_to_m = lambda x: x / 1000.0
sensor_coords_mm = {
    "L3-Z": [7817.60, -4635.30, 635.76], "L4-Z": [9707.20, -4635.30, 736.76],
    "L5-Z": [9021.68, -4533.70, 695.05], "L6-Z": [9377.47, -4533.70, 698.10],
    "L7-Z": [8562.14, -3822.50, 698.39], "L8-Z": [9248.77, -3822.50, 698.34],
    "L9-Z": [8120.68, -3060.50, 706.75], "L10-Z": [9100.08, -3060.50, 700.36],
    "L11-Z": [7436.69, -2019.10, 714.79], "L12-Z": [8886.55, -1993.70, 703.20],
    "L13-Z": [7483.04, -1087.78, 757.00], "L14-Z": [8369.19, -1085.70, 737.23],
    "L15-Z": [12540.50, -2552.19, 77.44], "L16-Z": [12839.15, -2436.23, -21.22],
    "L17-Z": [11030.13, -1027.66, 963.33], "L18-Z": [13002.27, -876.70, 193.41],
    "R3-Z": [7817.60, 4635.30, 635.76], "R4-Z": [9707.20, 4635.30, 736.76],
    "R5-Z": [9021.68, 4533.70, 695.05], "R6-Z": [9377.47, 4533.70, 698.10],
    "R7-Z": [8562.14, 3822.50, 698.39], "R8-Z": [9248.77, 3822.50, 698.34],
    "R9-Z": [8120.68, 3060.50, 706.75], "R10-Z": [9100.08, 3060.50, 700.36],
    "R11-Z": [7436.69, 2019.10, 714.79], "R12-Z": [8886.55, 1993.70, 703.20],
    "R13-Z": [7483.04, 1087.78, 757.00], "R14-Z": [8369.19, 1085.70, 737.23],
    "R15-Z": [12540.50, 2552.19, 77.44], "R16-Z": [12839.15, 2436.23, -21.22],
    "R17-Z": [11030.13, 1027.66, 963.33], "R18-Z": [13002.27, 876.70, 193.41],
}
dof_coords = np.array([mm_to_m(np.array(sensor_coords_mm[ch])) for ch in response_channels])

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
a = EMA.Model(H1_array, f, lower=3, upper=80, pol_order_high=60, driving_point=3, frf_type='accelerance')
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

# --- 3D Mode Shape Visualization ---
mode_shapes = a.normal_mode()
frequencies = a.nat_freq

spanwise = 12  # total front/rear pairs
chordwise = 2

sensor_indices = [
    [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],    # Left wing
    [26, 27], [24, 25], [22, 23], [20, 21], [18, 19], [16, 17]  # Right wing
]

for i in range(min(3, mode_shapes.shape[1])):
    mode = np.real(mode_shapes[:, i])
    scale = 0.5
    deformed = dof_coords.copy()
    deformed[:, 2] += scale * mode
    # Create structured grid (X, Y, Z)
    X, Y, Z = [], [], []
    for pair in sensor_indices:
        row_x = [deformed[pair[0], 0], deformed[pair[1], 0]]
        row_y = [deformed[pair[0], 1], deformed[pair[1], 1]]
        row_z = [deformed[pair[0], 2], deformed[pair[1], 2]]
        X.append(row_x)
        Y.append(row_y)
        Z.append(row_z)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='k')
    fig.colorbar(surf, ax=ax, label="Z Deflection")
    ax.set_title(f"Mode Shape {i+1} at {frequencies[i]:.2f} Hz")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    plt.tight_layout()
    plt.show()

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

a.A.shape
a.A[:, 0]
plt.plot(a.normal_mode()[:, :3]);

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




