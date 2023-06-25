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

N = 701
df = 1 / N

Fval = np.linspace(8.4, 9.6, 1000)
fit_interval = np.arange(7, 13, 1)

# desired output
CompareMethods = True   # Compares the methods in a graph
NoiseAnalysis = False   # Outputs a graph for the mean o fmultiple samples with noise.
PhaseAnalysis = True    # Compares for different phase shifts per method


# which methods do you want?
lt = ["Quadratic", "Barycentric", "Jains", "Quinns2nd"]
# lt = [lt[3]] 

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
    """
    Function that finds the true location of the peak using different methods.   
    """
    # we remove the first index, which is the offset
    data[0] = 0
    # We only need to do find the peak over the first half of the datase.
    peak_index = np.argmax(np.abs(data[:m.ceil(len(data)/2)]))
   
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
        resP = varyFrequency(Ndata, fData, Phases[i], rms_noise, method)
        Pres = np.append(Pres, resP)
    Pres = np.resize(Pres, (len(Phases), len(Fval)))
    return Pres.T

def getMeanandSTD(Ndata=700, fData=None, rms_noise=None,phase = 0, method="Quinns2nd", Iter = 100):
    """ Gets the mean and STD of multiple input signals with same phase/freqg but with random noise"""
    fp = np.full((len(Fval), Iter), np.nan)
    for i in range(Iter):
        fp[:,i] =  varyFrequency(Ndata, fData, phase, rms_noise, method)
    ffn = np.mean(fp, axis = 1)
    fvn = np.sqrt(np.var(fp, axis = 1))
    return np.vstack((ffn,fvn)).T


#%% functions for plotting

def peakFitplot(Ndata=700, fData=None, phase=0, rms_noise=None, method="Quinns2nd", plotmisfit = True, onlymisfits = False):
    " plots the peak peak fit for one method Can choose to only plot the peakfit or misfit."
    dummyvar = varyFrequency(Ndata, fData, phase, rms_noise, method)
    if plotmisfit:
        plt.figure(figsize = (10, 10))
        plt.subplot(211)
        plt.plot(fData, dummyvar, '.-',  label=method, linewidth = 0.5, markersize = 0.8)
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('peak fit (pxl)')
        plt.legend()
        plt.title("Fit of FFT Peak Positions")
        
        
        plt.subplot(212)
        plt.plot(fData, dummyvar - fData, '.-',  label=method, linewidth = 0.5, markersize = 0.8)
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('misfit')
        plt.legend()
        plt.title("Misfit of the FFT Peak Positions")
        plt.show()
        
    if plotmisfit == False:
        plt.figure(figsize = (10, 10))
        plt.plot(fData, dummyvar, '.-',  label=method, linewidth = 0.5, markersize = 0.8)
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('peak fit (pxl)')
        plt.legend(method)
        plt.title("Fit of FFT Peak Positions")
        
    if onlymisfits:
        plt.figure(figsize = (10, 10))
        plt.plot(fData, dummyvar - fData, '.-',  label=method, linewidth = 0.5, markersize = 0.8)
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('misfit')
        plt.legend(method)
        plt.title("Misfit of the FFT Peak Positions")
        plt.show()

        

def plotCompareMethods(Ndata=700, fData=None, phase=0, rms_noise=None, methods=["Quadratic", "Barycentric", "Jains", "Quinns2nd"], plotmisfit = True, onlymisfits = False):
    """ plots the peak peak fit for multiple methods, methods have to be given in a list of strings. 
        Can choose to only plot the peakfit or misfit. """
    ff = np.full((len(Fval), len(lt)), np.nan)
    for ii in range(len(methods)):
        ff[:, ii] = varyFrequency(Ndata, fData, phase, rms_noise, methods[ii])
    
    if plotmisfit:
        plt.figure(figsize = (10, 10))
        
        plt.subplot(211)
        for i in range(len(methods)):
            plt.plot(fData, ff[:,i], '.-',  label=methods[i], linewidth = 0.5, markersize = 0.8)
    
        plt.xlabel('Frequency (units of $\Delta$ f)' )
        plt.ylabel('peak fit (pxl)')
        plt.legend(methods)
        plt.title("Fit of FFT Peak Positions for different methods")
        
        plt.subplot(212)
        for i in range(len(methods)):
            plt.plot(fData, ff[:, i] - fData, '.-',  label=methods[i], linewidth = 0.5, markersize = 0.8)
      
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('misfit')
        plt.legend(methods)
        plt.show()
    
    if plotmisfit == False:
        plt.figure(figsize = (10, 10))

        for i in range(len(methods)):
            plt.plot(fData, ff[:,i] , '.-',  label=methods[i], linewidth = 0.5, markersize = 0.8)
    
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('peak fit (pxl)')
        plt.legend(methods)
        plt.title("Fit of FFT Peak Positions for different methods")
        plt.show()
    
    if onlymisfits:
        plt.figure(figsize = (10, 10))
        for i in range(len(methods)):
            plt.plot(fData, ff[:, i] - fData, '.-',  label=methods[i], linewidth = 0.5, markersize = 0.8)
      
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('misfit')
        plt.legend(methods)
        plt.show()
        
def MaxMisfit(Ndata=700, fData=None, phase=0, rms_noise=None, method="Quinns2nd", plotmisfit = True, onlymisfits = False):
    "returns the maximum deviation from the original input signal"
    ff = varyFrequency(Ndata, fData, phase, rms_noise, method) - fData
    if abs(min(ff)) > max(ff):
        return abs(min(ff))
    else:
        return max(ff)
        

def PlotPhaseDif(Ndata=700, fData=None, rms_noise=None, method="Quinns2nd", Phases = np.arange(0, 6)*0.2*np.pi, plotmisfit = True, onlymisfits = False):
    " Plots the peakfit for different phases in one graph, can choose to only plot the peakfit or misfit."
    ffP = varyPhase(Ndata, fData, rms_noise, method, Phases)
    if plotmisfit:
        plt.figure(figsize = (10, 10))
        plt.subplot(211)
        for i in range(len(Phase_shift)):
            plt.plot(fData, ffP[:,i], '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5, markersize = 0.8)
        plt.xlabel('Frequency (units of $\Delta$ f)',  size=23)
        plt.ylabel('peak fit (pxl)',  size=23)
        plt.legend(fontsize=15)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        # plt.title(f'Peakfit with Phase differences for {method}')


        plt.subplot(212)
        for i in range(len(Phase_shift)):
            plt.plot(fData, ffP[:,i] - fData, '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5, markersize = 0.8)
        plt.xlabel('Frequency (units of $\Delta$ f)',  size=23)
        plt.ylabel('misfit ()',  size=23)
        # plt.title(f'misfit of Peakfit with Phase differences for {method}')
        plt.legend(fontsize=15)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(f'phase {method}.pdf', format='pdf', dpi=300)
        plt.show()
        
    if plotmisfit == False:
        plt.figure()
        for i in range(len(Phase_shift)):
            plt.plot(fData, ffP[:,i], '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5, markersize = 0.8)
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('peak fit (pxl)')
        plt.legend()
        plt.title(f'Peakfit with Phase differences for {method}')
        plt.show()
    if onlymisfits:
        plt.figure()
        for i in range(len(Phase_shift)):
            plt.plot(fData, ffP[:,i] - fData, '.-', label = f'{Phase_shift[i]/np.pi: .2f} π ', linewidth = 0.5, markersize = 0.8)
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('misfit ()')
        plt.title(f'misfit of Peakfit with Phase differences for {method}')
        plt.legend()
        plt.show()

def plotNoiseAnalysis(Ndata=700, fData=None, rms_noise=RMS,phase = 0, method="Quinns2nd", Iter = 100, plotSTD = True, onlySTD = False):
    " plots the mean and STD of a set of input data with different Noises over it. " 
    NA = getMeanandSTD(Ndata, fData, rms_noise, phase, method, Iter)
    if plotSTD:
        plt.figure(figsize = (10, 10))
        plt.subplot(211)
        plt.plot(fData, NA[:,0], '.-' , linewidth = 0.5, markersize = 0.8)
        # plt.title(f' Average Fit of FFT Peak Position with {method}')
        plt.xlabel('Frequency (units of $\Delta$ f)',  size=23)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel('Average peak fit (pxl)',  size=23)
    
        plt.subplot(212)
        plt.plot(fData, np.sqrt(NA[:,1]), '.-' , linewidth = 0.5, markersize = 0.8)
        # plt.title(f"effect of noise {rms_noise}")
        plt.xlabel('Frequency (units of $\Delta$ f)',  size=24)
        plt.ylabel('std peak fit (units of $\Delta$ f)',  size=23)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(f'Noise {method} with {Iter} iterations and magnitude of {rms_noise} .pdf', format='pdf', dpi=300)
        plt.show()
    if plotSTD == False:
        plt.figure(figsize = (10, 10))
        plt.plot(fData, NA[:,0], '.-' , linewidth = 0.5, markersize = 0.8)
        plt.title(f' Average Fit of FFT Peak Position with {method}')
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('Average peak fit (pxl)')
        
        plt.show()
    if onlySTD:
        plt.figure()
        plt.plot(fData, np.sqrt(NA[:,1]), '.-' , linewidth = 0.5, markersize = 0.8)
        plt.title(f"effect of noise {rms_noise}")
        plt.xlabel('Frequency (units of $\Delta$ f)')
        plt.ylabel('std peak fit (units of $\Delta$ f)')
        plt.show()
          
    
#%% plots one data
a = generateData(Ndata=700, fData=8.5, phase=0, rms_noise=0.1, Gnoise=False)

plt.figure()
plt.plot(np.array(range(700))/700, a)
plt.xlabel('Time t',  size=14)
plt.ylabel('Amplitude',  size=14)
plt.savefig('input data w noise.pdf', format='pdf', dpi=300)

plt.show()

#%% maximum deviations

for i in lt:
    print(MaxMisfit(Ndata = N, fData = Fval, method= i) *100)


#%% check if everything is going well 

peakFitplot(Ndata = N, fData = Fval)
plotCompareMethods(Ndata = N, fData = Fval, methods= lt[2:4])

#%%
PlotPhaseDif(Ndata= N, fData= Fval, Phases = Phase_shift)
PlotPhaseDif(Ndata= N, fData= Fval, Phases = Phase_shift, method = "Jains")

#%%
# plotNoiseAnalysis(Ndata= N, fData= Fval)
# plotNoiseAnalysis(Ndata= N, fData= Fval, method ='Jains')

#%% noise, magnitude of the noise
"""
in this case the magnitude does clearly impact the STD. 
for greater magnitudes of STD the STD curve does get higher, and the peaks get 
a bit more spread out. this holds for both cases.
 """
# plotNoiseAnalysis(Ndata= N, fData= Fval, rms_noise=RMS , method = "Jains")
# plotNoiseAnalysis(Ndata= N, fData= Fval, rms_noise=1 , method = "Jains")

# plotNoiseAnalysis(Ndata= N, fData= Fval, rms_noise=RMS )
# plotNoiseAnalysis(Ndata= N, fData= Fval, rms_noise=1 )

# a = generateData(Ndata = N, fData = Fval)