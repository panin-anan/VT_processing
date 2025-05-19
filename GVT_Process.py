
#needed pip install sdypy
#needed pip install pyFRF
#needed pip install nptdms

from sdypy import EMA
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsWriter, ChannelObject
from nptdms import TdmsFile
import pyFRF


#Self Generated Input as .tdms
# Parameters
fs = 2000           # Sampling frequency [Hz]
f = 10              # Sine wave frequency [Hz]
T = 5               # Duration [s]
A = 1.0             # Amplitude [m/s^2]

# --- Generate signal ---
t = np.linspace(0, T, int(fs*T), endpoint=False)

# Acceleration signal with 10 Hz, 20 Hz, and 40 Hz components
acc = (
    1.0 * np.sin(2 * np.pi * 40 * t) +
    0.8 * np.sin(2 * np.pi * 100 * t) +
    0.6 * np.sin(2 * np.pi * 130 * t)
)

# Force input with a phase shift
force = (
    0.5 * np.sin(2 * np.pi * 40 * t + np.pi / 3) +
    0.4 * np.sin(2 * np.pi * 100 * t + np.pi / 4) +
    0.3 * np.sin(2 * np.pi * 130 * t + np.pi / 6)
)

# --- Save to .tdms ---
with TdmsWriter("example_signal.tdms") as writer:
    writer.write_segment([
        ChannelObject("TestGroup", "Time", t),
        ChannelObject("TestGroup", "Force", force),
        ChannelObject("TestGroup", "Acceleration", acc)
    ])


#loading .tdms

from nptdms import TdmsFile
from scipy.signal import csd

# --- Load data ---
tdms = TdmsFile.read("example_signal.tdms")
t = tdms["TestGroup"]["Time"].data
force = tdms["TestGroup"]["Force"].data
acc = tdms["TestGroup"]["Acceleration"].data
fs = 1 / (t[1] - t[0])

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

# --- Compute FRF from raw experimental data ---
RawData_FRF = pyFRF.FRF(fs, exc=force, resp=acc,exc_type='f', resp_type='a', frf_type='H1')


f, Pxy = csd(acc, force, fs=fs, nperseg=4096)
_, Pxx = csd(force, force, fs=fs, nperseg=4096)
H1 = Pxy / Pxx

# --- Save FRF to .npy for sdypy-EMA ---
H1_main = H1[None, None, :]  # shape becomes (1 input, 1 output, N freqs)
np.save("acc_data.npy", (f, H1_main))


#EMA processing
freqs, H1_main = np.load("acc_data.npy", allow_pickle=True)
selected_response = 1
frf = H1_main[:, selected_response, :]
a = EMA.Model(
    frf,
    freqs,
    lower=10,
    upper=200,
    pol_order_high=60,
    frf_type='accelerance'
    )

a.get_poles()
a.select_poles()

#As an alternative to selecting from the stabilisation chart, the frequencies can be defined as a list, e.g. (comment out to use)
#n_freq =  [176,476,932,1534,2258,3161,4180]
#a.select_closest_poles(n_freq)

frf_rec, modal_const = a.get_constants(whose_poles='own', FRF_ind='all', upper_r=False)

a.print_modal_data()

#plt.plot(a.normal_mode()[:, :3])

freq_a = a.freq
select_loc = 0

plt.figure(figsize = ((10,6)))
plt.subplot(211)

plt.semilogy(freqs, np.abs(frf[select_loc]), label='Experiment')
plt.semilogy(freq_a, np.abs(frf_rec[select_loc]),'--', label='LSCF')
plt.xlim(0,freqs[-1])
plt.ylabel(r"abs($\alpha$)")

plt.legend(loc = 'best')

plt.subplot(212)
plt.plot(freqs, np.angle(frf[select_loc],deg = 1), label='Experiment')
plt.plot(freq_a, np.angle(frf_rec[select_loc],deg = 1),'--',label='LSCF')
plt.xlim(0,freqs[-1])

plt.ylabel(r"angle($\alpha$)")
plt.legend(loc = 'best');

autoMAC = a.autoMAC()
plt.matshow(autoMAC)