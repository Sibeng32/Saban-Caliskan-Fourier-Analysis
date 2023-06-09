# -*- coding: utf-8 -*-
"""
Created on Mon May  8 00:05:23 2023

@author: saban
"""
import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.fftpack import fft, ifft
# from numpy.fft import fft

N = 700
df = 1 / N

Fval = np.linspace(8.4, 9.6, 1000)
fit_interval = np.arange(7, 13, 1)
fit_offset = m.ceil(len(fit_interval)/2) + 1/2

# desired output
CompareMethods = True   # Compares the methods in a graph
NoiseAnalysis = False   # Outputs a graph for the mean o fmultiple samples with noise.
PhaseAnalysis = True    # Compares for different phase shifts per method


# which methods do you want?
lt = ["Quadratic", "Barycentric", "Jains", "Quinns2nd"]
lt = [lt[3]] 

# input data
DataSimulation = True 
RealData = False 

# input for simulation:
Gnoise = False                # Does the signal have a gaussian curve to it?
DifferentPhase = True        # do you want to compare  for different phase shifts?

ShowInputsignal = False       # shows a graph of the first signal

# which phases do you want?
Phase_shift = np.arange(0, 6)*0.2*np.pi


Nnoise = 1000 # How many samples
RMS = 0.1 # Magnitude of Noise




#%% peakfit function
 
def FFT_peakFit(data, method): 
    # Finding the peak 
    peak_index = np.argmax(np.abs(data))
   
    # finding the Peak location using different methods:
    if method == "Quadratic": 
        y1 = np.abs(data[peak_index - 1])
        y2 = np.abs(data[peak_index])
        y3 = np.abs(data[peak_index + 1])
        d = (y3 - y1) / (2 * (2 * y2 - y1 - y3))
        k = peak_index + d
        return k
        
    elif method == "Barycentric": 

        y1 = np.abs(data[peak_index - 1])
        y2 = np.abs(data[peak_index])
        y3 = np.abs(data[peak_index + 1])
        d = (y3 - y1) / (y1 + y2 + y3)
        k_bar = peak_index + d
        return k_bar
    
    elif method == "Jains": 
        y1 = np.abs(data[peak_index - 1])
        y2 = np.abs(data[peak_index])
        y3 = np.abs(data[peak_index + 1])

        if y1 > y3:
            a = y2 / y1
            d = a / (1 + a)
            jains_coord = peak_index - 1 + d
        else:
            a = y3 / y2
            d = a / (1 + a)
            jains_coord = peak_index + d

        return jains_coord
    
    elif method == "Quinns2nd": 

        tau = lambda x: 1 / 4 * np.log10(3 * x ** 2 + 6 * x + 1) - np.sqrt(6) / 24 * np.log10((x + 1 - np.sqrt(2 / 3)) / (x + 1 + np.sqrt(2 / 3)))
        
        LoDe = ( data[peak_index].real * data[peak_index].real + data[peak_index].imag * 
                data[peak_index].imag) # long denominator
        
        ap = (data[peak_index + 1].real * data[peak_index].real + data[peak_index + 1].imag * 
              data[peak_index].imag) / LoDe
        dp = -ap / (1 - ap)
        
        am = (data[peak_index - 1].real * data[peak_index].real + data[peak_index - 1].imag *
              data[peak_index].imag) / LoDe
        dm = am / (1 - am)


        d = (dp + dm) / 2 + tau(dp * dp) - tau(dm * dm)
        
        quinns2nd_coord = peak_index + d

        return quinns2nd_coord       
        
        pass
    else:
        raise ValueError("Available methods are: 'Quadratic', 'Barycentric', 'Jains', and 'Quinns2nd'. Check spelling or update function")



#%%  data simulation functions

def generateData(Ndata=700, fData=9, phase=0, rms_noise=None, Gnoise=False):
    """Generate data of specified length, frequency, and -- if rms
    given -- with noise, and/or with a Gaussian form to the data if Gnoise = True."""
    if Gnoise == True:
        df = 1 / Ndata
        data = np.sin(2 * np.pi * fData * df * np.arange(1, Ndata + 1) + phase) + 2 * np.exp(
            -0.03 * df * (np.arange(1, Ndata + 1) - 200) ** 2)
        if rms_noise is not None:
            data += np.random.normal(scale=rms_noise, size=Ndata)
    elif Gnoise == False:
        df = 1/Ndata
        data = np.sin(2 * np.pi * fData * df * np.arange(1, Ndata + 1) + phase)
        if rms_noise is not None:
            data += np.random.normal(scale=rms_noise, size=Ndata)
    return data


def varyFrequency(Ndata=700, fData=None, phase=0, rms_noise=None, method="Quinns2nd"):
    """Generate fits with a range of frequencies"""
    NFreq = len(fData)
    res = np.zeros(NFreq)
    for ii in range(NFreq):
        res[ii] = FFT_peakFit(fft(generateData(Ndata, fData[ii], phase, rms_noise)), method)
    return res


def varyPhase(Ndata=700, fData=None, rms_noise=None, method="Quinns2nd", Phases = np.arange(0, 6)*0.2*np.pi):
    "Generates fits of a range of frequencies for different phases shifts"
    Pres = np.array([])
    for i in range(len(Phases)):
        resP = varyFrequency(Ndata, fData, Phases[i], rms_noise)
        Pres = np.append(Pres, resP)
    Pres = np.resize(Pres, (len(Phases), len(Fval)))
    return Pres.T


def getSTD(Ndata=700, fData=None, rms_noise=None,phase = 0, method="Quinns2nd", Iter = 100):
    """ Gets the mean and STD of multiple input signals with same phase/freqg but with random noise"""
    NA = np.array([])
    for i in range(Iter):
        fp =  varyFrequency(Ndata, fData, phase, rms_noise, method)
        NA = np.append(NA, fp)
    NA = (np.resize(NA, (Iter, len(Fval)))).T
    ffn = np.mean(NA, axis = 1)
    fvn = np.sqrt(np.var(NA, axis = 1))
    return np.vstack((ffn,fvn)).T
        

#%% check if everything is going well 
dummyvar = varyFrequency(Ndata= N, fData= Fval)
dummyvar2 = varyPhase(Ndata= N, fData= Fval, Phases = Phase_shift)
dummyvar3 = varyFrequency(Ndata= N, fData= Fval, rms_noise = RMS)
dumvar4 = getSTD(Ndata = N, fData = Fval, phase = 0, rms_noise= RMS, Iter = 10)


plt.figure()
plt.plot(dummyvar3)
plt.show()

plt.figure(figsize = (10, 10))
h1 = plt.subplot(211)
for i in range(len(Phase_shift)):
    plt.plot(Fval, dummyvar2[:, i] + 2*fit_offset, '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5, markersize = 0.8)
plt.xlabel('Frequency (units of $\Delta$ f)')
plt.ylabel('peak fit (pxl)')
plt.title(f'Phase differencces with {lt[l]}')


h2 = plt.subplot(212)
for i in range(len(Phase_shift)):
    plt.plot(Fval, (dummyvar2[:, i] + 2*fit_offset) - Fval, '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5, markersize = 0.8)
plt.xlabel('Frequency (units of $\Delta$ f)')
plt.ylabel('misfit ()')
plt.legend()
plt.show()


#%% first sample of input data

if ShowInputsignal == True:
    plt.figure(figsize = (10, 10))
    plt.plot(np.array(SInput[0]))
    plt.xlabel('Time')
    plt.ylabel('amplitude')
    plt.title('first sample of input signal}')
    plt.show

#%% Phase Analysis

if PhaseAnalysis:
    for l in range(len(lt)): 
        plt.figure(figsize = (10, 10))
        h1 = plt.subplot(211)
        
        for i in range(len(Phase_shift)):
            plt.plot(Fval, PA[:, i,l] + 2*fit_offset, '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5, markersize = 0.8)
        
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('peak fit (pxl)')
        plt.title(f'Phase differencces with {lt[l]}')
        plt.legend()
        
        h2 = plt.subplot(212)
        for i in range(len(Phase_shift)):
            plt.plot(Fval, (PA[:, i, l] + 2*fit_offset) - Fval, '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5, markersize = 0.8)
        
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('misfit ()')
        plt.legend()
        plt.show()

#%% Compare plots

sin = []

if CompareMethods:
    
    if DifferentPhase:
        
        for i in range(len(Phase_shift)): 
            plt.figure(figsize = (10, 10))
            h1 = plt.subplot(211)
            
            for l in range(len(lt)):
                plt.plot(Fval, PA[:, i, l] + 2*fit_offset, '.-',  label=lt[l], linewidth = 0.5, markersize = 0.8)
            
            plt.xlabel('Frequency (units of $\Delta$ f)')
            plt.ylabel('peak fit (pxl)')
            plt.legend()
            plt.title(f'Peaks using different methods with Phase shift of {Phase_shift[i]/np.pi: .2f} π')
        
            
            h2 = plt.subplot(212)
            for l in range(len(lt)):
                plt.plot(Fval, (PA[:, i, l] + 2*fit_offset) - Fval, '.-',  label=lt[l], linewidth = 0.5, markersize = 0.8)
        
            plt.xlabel('Frequency (units of $\Delta$ f)')
            plt.ylabel('misfit ()')
            plt.title(f' Missfit per peak per method with Phase shift of {Phase_shift[i]/np.pi: .2f} π')
    
            plt.legend()
            plt.show()
        
                
        
    if DifferentPhase == False:
        
        plt.figure(figsize = (10, 10))
        h1 = plt.subplot(211)
        for i in range(len(lt)):
            plt.plot(Fval, ff[:,i] + 2*fit_offset, '.-',  label=lt[i], linewidth = 0.5, markersize = 0.8)
    
        plt.plot(Fval, pm + 2*fit_offset, 'b', label='Maximum pixel')
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('peak fit (pxl)')
        plt.legend(lt + ['Maximum pixel'])
        
        h2 = plt.subplot(212)
        for i in range(len(lt)):
            plt.plot(Fval, (ff[:, i] + 2*fit_offset) - Fval, '.-',  label=lt[0], linewidth = 0.5, markersize = 0.8)
      
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('misfit')
        plt.legend(lt)
        plt.show()
    
#%% code for noise analysis

if NoiseAnalysis:
    
    plt.figure(figsize = (10, 10))
    h3 = plt.subplot(211)
    plt.plot(Fval, ffn+ 2*fit_offset, '.-' , linewidth = 0.5, markersize = 0.8)
    plt.title(' Average Fit of FFT Peak Position')
    plt.xlabel('Frequency (units of $\Delta$ f)')
    plt.ylabel('Average peak fit (pxl)')
    plt.legend(lt)
    
    h4 = plt.subplot(212)
    plt.plot(Fval, np.sqrt(fvn), '.-' , linewidth = 0.5, markersize = 0.8)
    plt.title(f"effect of noise {RMS}")
    plt.xlabel('Frequency (units of $\Delta$ f)')
    plt.ylabel('std peak fit (units of $\Delta$ f)')
    plt.legend(lt)
    plt.show()
    


