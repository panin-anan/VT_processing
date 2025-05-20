import numpy as np
from nptdms import TdmsWriter, ChannelObject
import os

# --- Ensure folder exists ---
folder = "testdata"
os.makedirs(folder, exist_ok=True)

# --- File path ---
tdms_path = os.path.join(folder, "example_signal.tdms")

# Parameters
fs = 2000           # Sampling frequency [Hz]
T = 5               # Duration [s]

# --- Generate signal ---
t = np.linspace(0, T, int(fs*T), endpoint=False)

# Acceleration signal with 40, 100, and 130 Hz components
acc = (
    1.0 * np.sin(2 * np.pi * 40 * t) +
    0.8 * np.sin(2 * np.pi * 100 * t) +
    0.6 * np.sin(2 * np.pi * 130 * t)
)

# Force input with phase shifts
force = (
    0.5 * np.sin(2 * np.pi * 40 * t + np.pi / 3) +
    0.4 * np.sin(2 * np.pi * 100 * t + np.pi / 4) +
    0.3 * np.sin(2 * np.pi * 130 * t + np.pi / 6)
)

# --- Save to .tdms ---
with TdmsWriter(tdms_path) as writer:
    writer.write_segment([
        ChannelObject("TestGroup", "Time", t),
        ChannelObject("TestGroup", "Force", force),
        ChannelObject("TestGroup", "Acceleration", acc)
    ])
