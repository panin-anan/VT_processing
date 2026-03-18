import re
import numpy as np
import tkinter as tk
from tkinter import filedialog

# =========================
# SELECT INPUT FILE
# =========================
root = tk.Tk()
root.withdraw()  # hide main window

f06_file = filedialog.askopenfilename(
    title="Select .f06 file",
    filetypes=[("F06 files", "*.f06"), ("All files", "*.*")]
)

if not f06_file:
    raise ValueError("No file selected")

# =========================
# SELECT OUTPUT FILE
# =========================
output_csv = filedialog.asksaveasfilename(
    title="Save CSV as",
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")]
)

if not output_csv:
    raise ValueError("No output file selected")

# =========================
# USER INPUT (AERO GRID)
# =========================
mach_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
k_list = [0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9]

n_modes = 20
n_blocks = len(mach_list) * len(k_list)

# =========================
# READ FILE
# =========================
with open(f06_file, 'r') as f:
    lines = f.readlines()

# =========================
# FIND QHHL MATRIX
# =========================
start_idx = None
for i, line in enumerate(lines):
    if "QHHL" in line and "MATRIX" in line:
        start_idx = i
        break

if start_idx is None:
    raise ValueError("QHHL matrix not found")

# =========================
# EXTRACT NUMBERS
# =========================
number_pattern = re.compile(r'[-+]?\d*\.\d+E[+-]?\d+')

data = []
for line in lines[start_idx:]:
    nums = number_pattern.findall(line)
    if nums:
        data.extend([float(x) for x in nums])

# =========================
# BUILD COMPLEX MATRIX
# =========================
complex_data = []
for i in range(0, len(data), 2):
    complex_data.append(complex(data[i], data[i+1]))

total_cols = n_modes * n_blocks
total_rows = n_modes

Q = np.array(complex_data[:total_rows * total_cols])
Q = Q.reshape((total_rows, total_cols))

# =========================
# WRITE CSV
# =========================
with open(output_csv, 'w') as f:

    for block in range(n_blocks):
        mach = mach_list[block // len(k_list)]
        k = k_list[block % len(k_list)]

        col_start = block * n_modes
        block_matrix = Q[:, col_start:col_start + n_modes]

        # Header
        f.write(f"Mach={mach}, k={k}\n")

        # Column headers
        f.write("," + ",".join([str(i+1) for i in range(n_modes)]) + "\n")

        # Matrix rows
        for i in range(n_modes):
            row_values = [
                f"{block_matrix[i,j].real:.6e}+{block_matrix[i,j].imag:.6e}j"
                for j in range(n_modes)
            ]
            f.write(f"{i+1}," + ",".join(row_values) + "\n")

        f.write("\n\n")

print("CSV written to:", output_csv)