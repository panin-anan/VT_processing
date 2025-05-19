
#needed pip install sdypy
#needed pip install pyFRF
#needed pip install nptdms

from sdypy import EMA
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsWriter, ChannelObject
from nptdms import TdmsFile
from pyFRF import FRF
import scipy
import pyExSi

# Define synthetic 3 DOF system
# degrees of freedom:
ndof = 3

# mass matrix:
m = 2
M = np.zeros((ndof, ndof))
np.fill_diagonal(M, m)

# stiffness matrix:
k = 4000000
K = np.zeros((ndof, ndof))

for i in range(K.shape[0] - 1):
    K[i,i] = 2*k
    K[i+1,i] = -k
    K[i,i+1] = -k
K[ndof-1,ndof-1] = k

# damping matrix:
c = 50
C = np.zeros((ndof, ndof))

for i in range(C.shape[0] - 1):
    C[i,i] = 2*c
    C[i+1,i] = -c
    C[i,i+1] = -c
C[ndof-1,ndof-1] = c

# eigenfrequencies:
eig_val, eig_vec = scipy.linalg.eigh(K, M)
eig_val.sort()
eig_omega = np.sqrt(np.abs(np.real(eig_val)))
eig_freq = eig_omega / (2 * np.pi)

# frequencies:
df = 0.5
freq_syn = np.arange(0.0, 2000, df)
omega = 2 * np.pi * freq_syn

# time:
T = 1 / (freq_syn[1] - freq_syn[0])
dt = 1 / (2*freq_syn[-1])
t = np.linspace(0, T, 2*len(freq_syn)-2)

# synthetic FRF matrix of the system:
FRF_matrix = np.zeros([M.shape[0], M.shape[0], len(freq_syn)], dtype="complex128")  # full system 3x3 FRF matrix
for i, omega_i in enumerate(omega):
    FRF_matrix[:,:,i] = scipy.linalg.inv(K - omega_i**2 * M + 1j*omega_i*C)

H1_syn = FRF_matrix[0,0,:]

fig, ax1 = plt.subplots()

ax1.semilogy(freq_syn, np.abs(H1_syn), 'b')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('H1', color='b')
ax1.set_xlim(left=0, right=1000)
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(freq_syn, np.angle(H1_syn), 'r', alpha=0.2)
ax2.set_ylabel('Angle', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')


'''
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
'''