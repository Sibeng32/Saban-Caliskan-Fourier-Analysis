# -*- coding: utf-8 -*-
"""
Onderzoeksstage: Fourier Analysis in Optical Coherence Tomography
Saban Caliskan 
'oefencode'
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fftpack import fft, ifft
# from numpy.fft import fft

# sampling rate
Ndata = 701
# sample interval 
t = np.arange(0,5,1/Ndata)

# sinusoide function with different frequencies and amplitudes
freq = [4]
Amp = [1]

x = []
 
for i in range(len(freq)):
    a = Amp[i]*np.sin(2*np.pi*freq[i]*t) * np.exp(-0.05*(t**2)) + 1.2*np.exp(-(t-1)**2)
    x.append(a)

x = sum(x)
x = np.array(x)

# plot of the function
plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')
plt.show()

#%% plot of  FFT

# FFT from data
F = fft(x)

F_abs = np.abs(F)
F_abs = F_abs[1:]

#%%


n = np.arange(len(F))
T = len(F)/Ndata
freq = n/T 
fx = scipy.fft.fftfreq(x.size, d=1/Ndata)

dum = np.vstack((fx,F_abs)).T
dummy = np.flip( dum[dum[:,0].argsort()[::-1]], 0)

# plots of both FFT and IFFT
plt.figure(figsize = (12, 6))

plt.subplot(1, 2, 1)
plt.title(f" FFT vlaues of a sinus with {Ndata} bins")
plt.bar(fx, F_abs, width = 1)

# plt.hist(dummy[:,1], bins= Ndata)

plt.xlabel('Freq ')
plt.ylabel('Amplitude')
plt.xlim(0, 10)

plt.subplot(1, 2, 2)
plt.title(f" {Ndata} data points of a sinus ")
plt.plot(t, ifft(F), 'r.', markersize=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

#%%
# # tried putting the values in bins 

# Bins = np.histogram(F_abs, bins=680)

# plt.figure()
# plt.stem(, Bins[0], 'b', markerfmt=" ", basefmt="-b")
# plt.xlim(0, 10)
plt.show()
