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

Fval = np.linspace(8.4, 9.6, 99)
fit_interval = np.arange(7, 13, 1)
fit_offset = m.ceil(len(fit_interval)/2) + 1/2

# desired output
CompareMethods = True   # Compares the methods in a graph
noiseAnalysis = True    # Outputs a graph for the mean o fmultiple samples with noise.
PhaseAnalysis = True    # Compares for different phase shifts per method

# which methods do you want?
lt = ["Quadratic", "Barycentric", "Jains", "Quinns2nd"]
# lt = [lt[2], lt[3]] 

# input data
DataSimulation = True 
RealData = False 

# input for simulation:
Gnoise = True                # Does the signal have a gaussian curve to it?
DifferentPhase = True        # do you want to compare  for different phase shifts?

# which phases do you want?
Phase_shift = np.arange(0, 10)*0.2*np.pi
Phase_shift = Phase_shift[0:5] # the other half produces the same output.

Nnoise = 1000 # How many samples
RMS = 0.1 # Magnitude of Noise




#%%
 
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



#%% Simulation Dataset

if DataSimulation == True:
    if Gnoise == True:
        if DifferentPhase:
            PA = np.full((len(Fval), len(Phase_shift), len(lt)), np.nan)
            for l in range(len(lt)):    
                for k in range(len(Phase_shift)):
                    ff = np.full((len(Fval), len(lt)), np.nan)
                    pm = np.full((len(Fval)), np.nan)
                
                    for i in range(len(Fval)):
                        y = np.sin(2 * np.pi * Fval[i] * df * np.arange(1, N+1) + Phase_shift[k]) * np.exp(-0.01* RMS*df*(np.arange(1, N+1) - 200)**2) + 2*np.exp(-0.03*df*(np.arange(1, N+1)- 200)**2)
                        fy = fft(y)
                        data = fy[fit_interval]
                        pm[i] = np.argmax(abs(data))
                        PA[i,k,l] = FFT_peakFit(data, lt[l])
        
        if DifferentPhase == False:
            
            ff = np.full((len(Fval), len(lt)), np.nan)
            fv = np.full((len(Fval), len(fit_interval)), np.nan)
            pm = np.full((len(Fval)), np.nan)

            for i in range(len(Fval)):
                y = np.sin(2 * np.pi * Fval[i] * df * np.arange(1, N+1)) * np.exp(-0.01* RMS*df*(np.arange(1, N+1) - 200)**2) + 2*np.exp(-0.03*df*(np.arange(1, N+1)- 200)**2)
                fy = fft(y)
                data = fy[fit_interval]
                fv[i, :] = data
                pm[i] = np.argmax(abs(data))

                for j in range(len(lt)):
                    ff[i, j] = FFT_peakFit(data, lt[j])
                    
    if Gnoise == False:
        
        if DifferentPhase:
            PA = np.full((len(Fval), len(Phase_shift), len(lt)), np.nan)
            for l in range(len(lt)):    
                for k in range(len(Phase_shift)):
                    ff = np.full((len(Fval), len(lt)), np.nan)
                    pm = np.full((len(Fval)), np.nan)
                
                    for i in range(len(Fval)):
                        y = np.sin(2 * np.pi * Fval[i] * df * np.arange(1, N+1) + Phase_shift[k])
                        fy = fft(y)
                        data = fy[fit_interval]
                        pm[i] = np.argmax(abs(data))
                        PA[i,k,l] = FFT_peakFit(data, lt[l])
                        
        if DifferentPhase == False:
            
            ff = np.full((len(Fval), len(lt)), np.nan)
            fv = np.full((len(Fval), len(fit_interval)), np.nan)
            pm = np.full((len(Fval)), np.nan)

            for i in range(len(Fval)):
                y = np.sin(2 * np.pi * Fval[i] * df * np.arange(1, N+1))
                fy = fft(y)
                data = fy[fit_interval]
                fv[i, :] = data
                pm[i] = np.argmax(abs(data))

                for j in range(len(lt)):
                    ff[i, j] = FFT_peakFit(data, lt[j])
            
    
#%%

if PhaseAnalysis:
    for l in range(len(lt)): 
        plt.figure(figsize = (10, 10))
        h1 = plt.subplot(211)
        
        for i in range(len(Phase_shift)):
            plt.plot(Fval, PA[:, i,l] + 2*fit_offset, '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5)
        
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('peak fit (pxl)')
        plt.title(f'Phase differencces with {lt[l]}')
        plt.legend()
        
        h2 = plt.subplot(212)
        for i in range(len(Phase_shift)):
            plt.plot(Fval, (PA[:, i, l] + 2*fit_offset) - Fval, '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5)
        
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
                plt.plot(Fval, PA[:, i, l] + 2*fit_offset, '.-',  label=lt[l], linewidth = 0.5)
            
            plt.xlabel('Frequency (units of $\Delta$ f)')
            plt.ylabel('peak fit (pxl)')
            plt.legend()
            plt.title(f'Peaks using different methods with Phase shift of {Phase_shift[i]/np.pi: .2f} π')
        
            
            h2 = plt.subplot(212)
            for l in range(len(lt)):
                plt.plot(Fval, (PA[:, i, l] + 2*fit_offset) - Fval, '.-',  label=lt[l],  linewidth = 0.5)
        
            plt.xlabel('Frequency (units of $\Delta$ f)')
            plt.ylabel('misfit ()')
            plt.title(f' Missfit per peak per method with Phase shift of {Phase_shift[i]/np.pi: .2f} π')
    
            plt.legend()
            plt.show()
        
                
        
    if DifferentPhase == False:
        
        plt.figure(figsize = (10, 10))
        h1 = plt.subplot(211)
        for i in range(len(lt)):
            plt.plot(Fval, ff[:,i] + 2*fit_offset, '.-',  label=lt[i], linewidth = 0.5)
    
        plt.plot(Fval, pm + 2*fit_offset, 'b', label='Maximum pixel')
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('peak fit (pxl)')
        plt.legend(lt + ['Maximum pixel'])
        
        h2 = plt.subplot(212)
        for i in range(len(lt)):
            plt.plot(Fval, (ff[:, i] + 2*fit_offset) - Fval, '.-',  label=lt[0], linewidth = 0.5)
      
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('misfit')
        plt.legend(lt)
        plt.show()
    
#%% code for noise analysis

if noiseAnalysis:
    ff = np.full((len(Fval), len(lt)), np.nan)
    fn = np.full((len(Fval), len(lt)), np.nan)
    fv = np.full((len(Fval), Nnoise), np.nan)
    
    for i in range(len(Fval)):
        y = np.sin(2 * np.pi * Fval[i] * df * np.arange(1, N+1))
        fy = fft(y + RMS * np.random.randn(Nnoise, N), axis=1)
        data = fy[:, fit_interval]
        
        for j in range(len(lt)):
            fd = []
            for k in range(len(data)):
                 fd.append(FFT_peakFit(data[k], lt[j]))
            ff[i, j] = np.mean(fd)
            fv[i, j] = np.var(fd)

    plt.figure(figsize = (10, 10))
    h3 = plt.subplot(211)
    plt.plot(Fval, ff+ 2*fit_offset, '.-')
    plt.title('Fit of FFT peak position')
    plt.xlabel('Frequency (units of $\Delta$ f)')
    plt.ylabel('Average peak fit (pxl)')
    plt.legend(lt)
    
    h4 = plt.subplot(212)
    plt.plot(Fval, np.sqrt(fv), '.-')
    plt.title(f"effect of noise {RMS}")
    plt.xlabel('Frequency (units of $\Delta$ f)')
    plt.ylabel('variance peak fit (units of $\Delta$ f)')
    plt.legend(lt)
    plt.show()

