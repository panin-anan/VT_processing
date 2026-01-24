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
        "L15-Z", "L16-Z",
        "L17-Z", "L18-Z",
        "R15-Z", "R16-Z",
        "R17-Z", "R18-Z", 
        #"V1-Y", "V2-Y", "V3-Y", "V4-Y"
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
    # Left Wing
    #"L3-Z": [7817.60, -4635.30, 635.76], "L4-Z": [9707.20, -4635.30, 736.76],
    #"L5-Z": [9021.68, -4533.70, 695.05], "L6-Z": [9377.47, -4533.70, 698.10],
    #"L7-Z": [8562.14, -3822.50, 698.39], "L8-Z": [9248.77, -3822.50, 698.34],
    #"L9-Z": [8120.68, -3060.50, 706.75], "L10-Z": [9100.08, -3060.50, 700.36],
    #"L11-Z": [7436.69, -2019.10, 714.79], "L12-Z": [8886.55, -1993.70, 703.20],
    #"L13-Z": [7483.04, -1087.78, 757.00], "L14-Z": [8369.19, -1085.70, 737.23],
    #H-Stab Left
    "L15-Z": [12540.50, -2552.19, 77.44], "L16-Z": [12839.15, -2436.23, -21.22],
    "L17-Z": [11030.13, -1027.66, 963.33], "L18-Z": [13002.27, -876.70, 193.41],
    #Right Wing
    #"R3-Z": [7817.60, 4635.30, 635.76], "R4-Z": [9707.20, 4635.30, 736.76],
    #"R5-Z": [9021.68, 4533.70, 695.05], "R6-Z": [9377.47, 4533.70, 698.10],
    #"R7-Z": [8562.14, 3822.50, 698.39], "R8-Z": [9248.77, 3822.50, 698.34],
    #"R9-Z": [8120.68, 3060.50, 706.75], "R10-Z": [9100.08, 3060.50, 700.36],
    #"R11-Z": [7436.69, 2019.10, 714.79], "R12-Z": [8886.55, 1993.70, 703.20],
    #"R13-Z": [7483.04, 1087.78, 757.00], "R14-Z": [8369.19, 1085.70, 737.23],
    #H-stab Right
    "R15-Z": [12540.50, 2552.19, 77.44], "R16-Z": [12839.15, 2436.23, -21.22],
    "R17-Z": [11030.13, 1027.66, 963.33], "R18-Z": [13002.27, 876.70, 193.41],
    #V-stab
    #"V1-Y": [11193.92, 47.61, 2078.57], "V2-Y": [11850.28, 57.65, 1811.22],
    #"V3-Y": [12289.34, 21.06, 3573.60], "V4-Y": [12607.14, 22.71, 3573.60],
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
a = EMA.Model(H1_array, f, lower=3, upper=80, pol_order_high=100, driving_point=3, frf_type='accelerance')
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

# ---------------------- Helper indexing by channel name ----------------------
def idx(ch):
    return response_channels.index(ch)

# ---------------------- Wing Mode Shape Visualization (L & R surfaces) ----------------------
# Build wing pairs by channel names (each row is a pair: [rear, front] along chord at a span station)
left_wing_pairs = [
    ("L15-Z", "L16-Z"),
    ("L17-Z", "L18-Z"),
]
right_wing_pairs = [
    ("R15-Z", "R16-Z"),
    ("R17-Z", "R18-Z"),
]

# Assemble rows (spanwise) as indices
sensor_indices = [[idx(a), idx(b)] for (a, b) in left_wing_pairs] + [[idx(a), idx(b)] for (a, b) in right_wing_pairs]

for i in range(min(10, mode_shapes.shape[1])):
    mode = np.real(mode_shapes[:, i])
    scale = 0.5
    deformed = dof_coords.copy()
    # Z-deflection for Z-DOF sensors; for Y-DOF (V*-Y) we’ll treat separately
    deformed[:, 2] += scale * mode

    # Create structured grid (X, Y, Z)
    X, Y, Z = [], [], []
    for pair in sensor_indices:
        row_x = [deformed[pair[0], 0], deformed[pair[1], 0]]
        row_y = [deformed[pair[0], 1], deformed[pair[1], 1]]
        row_z = [deformed[pair[0], 2], deformed[pair[1], 2]]
        X.append(row_x); Y.append(row_y); Z.append(row_z)

    X = np.array(X); Y = np.array(Y); Z = np.array(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='k')
    fig.colorbar(surf, ax=ax, label="Z Deflection")
    ax.set_title(f"Wing Mode Shape {i+1} at {frequencies[i]:.2f} Hz")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
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

autoMAC = EMA.tools.MAC(a.A, a.A)
plt.matshow(np.abs(autoMAC), cmap="viridis")
plt.colorbar(label='MAC Value')
plt.show()

# ================== UNV EXPORT (Uz only, DS-15 nodes) — SINGLE BLOCK ==================
# Requires: pip install pyuff
# Assumes these already exist from your script:
# - file_path (path of the input data file you processed)
# - response_channels (list of channel names in order)
# - dof_coords (Nx3 coordinates array for those channels, in meters unless you scale below)
# - a.normal_mode() and a.nat_freq from sdypy.EMA (mode shapes & frequencies)

# ================== UNV EXPORT (DS-15 nodes + DS-55 modes; Uz-only) ==================
# Modifications from your block:
# - DS-15 uses DENSE arrays aligned with node_nums (not keyed by node id).
# - DS-55 written per SDRL spec: 3-DOF translation vector declared (NDV=3),
#   Specific Data Type = 8 (Displacement), Data Type = 2 (Real).
# - r1/r2 zeros, r3 = Uz; arrays are DENSE (len == len(node_nums)).
# - No DS-164 (units). Set the correct Units in NVH or scale LENGTH_SCALE here.

import os
import numpy as np
import pyuff

# ---- Output folder (fixed) ----
save_dir = r"C:\Nin Folder\RVC\IRIS-T\GVT Processing\src\VT_processing\MAC_Compare_File"
os.makedirs(save_dir, exist_ok=True)
base_name = os.path.splitext(os.path.basename(file_path))[0]
unv_path = os.path.join(save_dir, f"{base_name}_exp_modes_Stab.unv")

# ---- Build node IDs (contiguous 1..N) ----
Nch = len(response_channels)
node_ids = np.arange(1, Nch + 1, dtype=int)  # 1..N

# ---- OPTIONAL length scaling (set to 1000.0 if your FE model is in mm) ----
LENGTH_SCALE = 1.0  # meters -> meters. Use 1000.0 for meters -> millimeters

# ---- DS-15 geometry (DENSE arrays aligned with node_nums) ----
X = (dof_coords[:, 0] * LENGTH_SCALE).astype(float)
Y = (dof_coords[:, 1] * LENGTH_SCALE).astype(float)
Z = (dof_coords[:, 2] * LENGTH_SCALE).astype(float)

export_cs = np.zeros_like(node_ids, dtype=int)  # 0 = global
def_cs    = np.zeros_like(node_ids, dtype=int)  # 0 = global
disp_cs   = np.zeros_like(node_ids, dtype=int)  # 0 = global
color     = np.zeros_like(node_ids, dtype=int)  # 0 = default

# Use explicit dict to be compatible across pyuff versions
ds15 = {
    "type": 15,
    "node_nums": node_ids,   # list/array of node ids present
    "x": X, "y": Y, "z": Z,  # DENSE arrays, same length as node_nums
    "export_cs": export_cs, "def_cs": def_cs, "disp_cs": disp_cs, "color": color
}

# ---- Experimental mode shapes (Uz only) ----
mode_shapes = a.normal_mode()         # shape: (Nch, Nmodes)
frequencies = np.asarray(a.nat_freq)  # Hz

uff_sets = [ds15]

for m_idx in range(mode_shapes.shape[1]):
    # Real-valued modal vector at measured DOFs
    phi = np.real(mode_shapes[:, m_idx]).astype(float)  # (Nch,)
    # DENSE component arrays per node (length == len(node_ids))
    r1 = np.zeros_like(phi)  # Ux
    r2 = np.zeros_like(phi)  # Uy
    r3 = phi                 # Uz

    # Prefer prepare_55 (enforces SDRL record layout). Fall back to dict if needed.
    try:
        ds55 = pyuff.prepare_55(
            # ID lines (helpful labels)
            id1=f"Mode {m_idx+1}",
            id2="EMA",
            id3="Acceleration shapes (Uz only)",
            id4="LoadCase 1",
            id5="",

            # Data definition (SDRL DS-55)
            model_type=1,           # 1 = Structural
            analysis_type=2,        # 2 = Normal modes
            data_ch=2,              # 2 = 3-DOF translation vector (UX,UY,UZ)
            spec_data_type=12,       # 12 = Acceleration
            data_type=2,            # 2 = Real
            n_data_per_node=3,      # NDV = 3 (components per node)

            # Mode info
            load_case=1,
            mode_n=m_idx + 1,
            freq=float(frequencies[m_idx]),
            modal_m=0.0,
            modal_damp_vis=0.0,
            modal_damp_his=0.0,

            # Per-node data (dense, aligned with node_nums)
            node_nums=node_ids.astype(int),
            r1=r1, r2=r2, r3=r3
        )
    except Exception:
        # Fallback dict for older pyuff variants
        ds55 = {
            "type": 55,
            "id1": f"Mode {m_idx+1}",
            "id2": "EMA",
            "id3": "Acceleration shapes (Uz only)",
            "id4": "LoadCase 1",
            "id5": "",
            "model_type": 1,
            "analysis_type": 2,
            "data_ch": 2,            # 3-DOF translation vector
            "spec_data_type": 12,     # Acceleration
            "data_type": 2,          # Real
            "n_data_per_node": 3,    # NDV = 3
            "load_case": 1,
            "mode_n": m_idx + 1,
            "freq": float(frequencies[m_idx]),
            "modal_m": 0.0,
            "modal_damp_vis": 0.0,
            "modal_damp_his": 0.0,
            "node_nums": node_ids,
            "r1": r1, "r2": r2, "r3": r3
        }

    uff_sets.append(ds55)

# ---- Write UNV robustly (constructor with filename first) ----
try:
    uff = pyuff.UFF(unv_path)
    uff.write_sets(uff_sets)
except TypeError:
    uff = pyuff.UFF()
    uff.write_sets(uff_sets, unv_path)

print(f"\nUNV written (DS-15 nodes + DS-55 modes; Uz-only in 3-DOF vector):\n{unv_path}\n"
      "In NVH MAC: set Custom DOF = UZ, map by coordinates, and set Units/tolerance appropriately.")
# ================== END UNV EXPORT ==================



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




