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
fit_offset = m.ceil(len(fit_interval) / 2) + 1
Nnoise = 1000
RMS = 0.1

Phase = True
Noise = False

lt = ["Quadratic", "Barycentric", "Jains", "Quinns2nd"]
lt = [lt[2], lt[3]]


Phases = np.linspace(1,10, 4)*0.2*np.pi

#%% FFT_peakfit function

def FFT_peakFit(data, method):
    # Finding the peak 
    x = np.arange(len(data))
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
    
        
        LoDe = ( data[peak_index].real * data[peak_index].real + data[peak_index].imag * data[peak_index].imag) # long denominator
        
        ap = (data[peak_index + 1].real * data[peak_index].real + data[peak_index + 1].imag * data[peak_index].imag) / LoDe
        dp = -ap / (1 - ap)
        
        am = (data[peak_index - 1].real * data[peak_index].real + data[peak_index - 1].imag * data[peak_index].imag) / LoDe
        dm = am / (1 - am)
        tau = lambda x: 1 / 4 * np.log(3 * x ** 2 + 6 * x + 1) - np.sqrt(6) / 24 * np.log(
            (x + 1 - np.sqrt(2 / 3)) / (x + 1 + np.sqrt(2 / 3)))

        d = (dp + dm) / 2 + tau(dp * dp) - tau(dm * dm)
        quinns2nd_coord = peak_index + d

        return quinns2nd_coord       
        
        pass
    else:
        raise ValueError("Available methods are: 'Quadratic', 'Barycentric', 'Jains', and 'Quinns2nd'. Check spelling or update function")

#%% Sinusoidal functon generator:

    
if Phase == True:
    for phi in Phases:
        bruh = phi
        if Noise == True:
            Total_sin =  []
            
            ff = np.full((len(Fval), len(lt)), np.nan)
            fn = np.full((len(Fval), len(lt)), np.nan)
            fv = np.full((len(Fval), Nnoise), np.nan)
     
            for i in range(len(Fval)):
                y = np.sin(2 * np.pi * Fval[i] * df * np.arange(1, N+1))
                fy = fft(y + RMS * np.random.randn(Nnoise, N), axis=1)
                data = fy[:, fit_interval]
                Total_sin.append(y + RMS * np.random.randn(Nnoise, N))
                
                for j in range(len(lt)):
                    fd = []
                    for k in range(len(data)):
                         fd.append(FFT_peakFit(data[k], lt[j]))
                    ff[i, j] = np.mean(fd)
                    fv[i, j] = np.var(fd)

            #plot sinus met ruis en faseverschil
            Total_sin = np.array(Total_sin)
            Total_sin = np.sum(np.array(Total_sin), 1)
            
            MeanSin = np.mean(Total_sin, axis = 1)
            plt.figure(figsize = (8, 6))
            plt.plot(MeanSin)
            plt.ylabel(f'Mean Amplitude with Noise met fase verschil {bruh}')
            plt.show()
            
            #plot DFFT van Sinus met fase verschil en ruis
            plt.figure(figsize = (10, 10))
            h3 = plt.subplot(211)
    
            plt.plot(Fval, ff, '.-')
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
            
        if Noise == False:
            Total_sin =  []
            ff = np.full((len(Fval), len(lt)), np.nan)
            fv = np.full((len(Fval), len(fit_interval)), np.nan)
            pm = np.full((len(Fval)), np.nan)
            
            
            for i in range(len(Fval)):
                y = np.sin(2 * np.pi * Fval[i] * df * np.arange(1, N+1) + phi )
                Total_sin.append(y)
                
                fy = fft(y)
                data = fy[fit_interval]
                fv[i, :] = data
                pm[i] = np.argmax(abs(data))
                
                for j in range(len(lt)):
                    ff[i, j] = FFT_peakFit(data, lt[j])
                
            # plot sinus met fase verschil, zonder ruis
            Total_sin = np.array(Total_sin)
            Total_sin = np.sum(np.array(Total_sin), 1)
            plt.figure(figsize = (8, 6))
            plt.plot(Total_sin )
            plt.ylabel(f'Amplitude met fase verschil {bruh}')
            plt.show()
        

            plt.figure(figsize = (10, 10))
            h1 = plt.subplot(211)
            for i in range(len(lt)):
                plt.plot(Fval, ff[:, i] + fit_offset, '.-',  label=lt[0], linewidth = 0.5)

            plt.plot(Fval, pm + fit_offset, 'b', label='Maximum pixel')
            plt.xlabel('Frequency (units of $\Delta$ f)')
            plt.ylabel('peak fit (pxl)')
            plt.legend(lt + ['Maximum pixel'])
            
            h2 = plt.subplot(212)
            for i in range(len(lt)):
                plt.plot(Fval, (ff[:, i] + fit_interval[1] - 2) - Fval, '.-',  label=lt[0], linewidth = 0.5)
          
            plt.xlabel('Frequency (units of $\Delta$ f)')
            plt.ylabel('misfit ()')
            plt.legend(lt)
            plt.show()
    

if Phase == False:

    if Noise == True:
        Total_sin =  []
        for i in range(len(Fval)):
            y = np.sin(2 * np.pi * Fval[i] * df * np.arange(1, N+1)) + RMS * np.random.randn(Nnoise, N)
            Total_sin.append(y)
            
        Total_sin = np.array(Total_sin)
        Total_sin = np.sum(np.array(Total_sin), 1)
        
        MeanSin = np.mean(Total_sin, axis = 1)
        plt.figure(figsize = (8, 6))
        plt.plot(MeanSin)
        plt.ylabel('Mean Amplitude with Noise')
        plt.show()
        
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
        
        plt.plot(Fval, ff, '.-')
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
            
        
    if Noise == False:
        Total_sin =  []
        for i in range(len(Fval)):
            y = np.sin(2 * np.pi * Fval[i] * df * np.arange(1, N+1))
            Total_sin.append(y)
        
        Total_sin = np.array(Total_sin)
        Total_sin = np.sum(np.array(Total_sin), 1)
        plt.figure(figsize = (8, 6))
        plt.plot(Total_sin )
        plt.ylabel('Amplitude')
        plt.show()
        
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

        plt.figure(figsize = (10, 10))
        h1 = plt.subplot(211)
        for i in range(len(lt)):
            plt.plot(Fval, ff[:, i] + fit_offset, '.-',  label=lt[0], linewidth = 0.5)

        plt.plot(Fval, pm + fit_offset, 'b', label='Maximum pixel')
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('peak fit (pxl)')
        plt.legend(lt + ['Maximum pixel'])
        
        h2 = plt.subplot(212)
        for i in range(len(lt)):
            plt.plot(Fval, (ff[:, i] + fit_interval[1] - 2) - Fval, '.-',  label=lt[0], linewidth = 0.5)
      
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('misfit ()')
        plt.legend(lt)
        plt.show()
        
