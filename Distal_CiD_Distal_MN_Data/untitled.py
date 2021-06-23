import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks
import matplotlib.colors as clr
from matplotlib.ticker import AutoMinorLocator

VLMN_1 = np.load('VLMN_run1_all_distal.npy')                                                                                                                          
VLMN_2 = np.load('VLMN_run2_all_distal.npy')                                                                                                                        
VLMN_3 = np.load('VLMN_run3_all_distal.npy')
VLMN_4 = np.load('VLMN_run4_all_distal.npy')
VLMN_5 = np.load('VLMN_run5_all_distal.npy')
VLMN_6 = np.load('VLMN_run6_all_distal.npy')
VLMN_7 = np.load('VLMN_run7_all_distal.npy')
VLMN_8 = np.load('VLMN_run8_all_distal.npy')
VLMN_9 = np.load('VLMN_run9_all_distal.npy')
VLMN_10 = np.load('VLMN_run10_all_distal.npy')
VLMN_11 = np.load('VLMN_run11_all_distal.npy')
VLMN_12 = np.load('VLMN_run12_all_distal.npy')
VLMN_13 = np.load('VLMN_run13_all_distal.npy')
VLMN_14 = np.load('VLMN_run14_all_distal.npy')
VLMN_15 = np.load('VLMN_run15_all_distal.npy')

Time = np.load('Time_run1_all_distal.npy')

#VLMN = np.concatenate((VLMN_1,VLMN_2,VLMN_3,VLMN_4,VLMN_5,VLMN_6,VLMN_7,VLMN_8,VLMN_9,VLMN_10,VLMN_11,VLMN_12,VLMN_13,VLMN_14,VLMN_15))

def TBF_Estimate(neuron_type,neuron_list,windowmin,windowmax):
    
    spikes_ind = []
    
    for neuron in np.arange(len(neuron_type)):
        thres=0
        spikest=np.where(neuron_type[neuron,:]>thres)
        spikes_indt = spikest[0]/10
    
        spikes_ind.append(spikes_indt)
    
    selectnrnspikes = []
    spikebins = []
    
    for nrn in neuron_list:
        indext = np.where(np.logical_and(spikes_ind[nrn]>windowmin, spikes_ind[nrn]<windowmax))
        
        spikebins.append(np.diff(spikes_ind[nrn][indext]))
        
    return spikebins

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_1))
    spikebins = TBF_Estimate(VLMN_1,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

Title = 'Runthrough'
spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_2))
    spikebins = TBF_Estimate(VLMN_2,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_3))
    spikebins = TBF_Estimate(VLMN_3,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_4))
    spikebins = TBF_Estimate(VLMN_4,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_5))
    spikebins = TBF_Estimate(VLMN_5,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_6))
    spikebins = TBF_Estimate(VLMN_6,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_7))
    spikebins = TBF_Estimate(VLMN_7,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_8))
    spikebins = TBF_Estimate(VLMN_8,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_9))
    spikebins = TBF_Estimate(VLMN_9,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_10))
    spikebins = TBF_Estimate(VLMN_10,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_11))
    spikebins = TBF_Estimate(VLMN_11,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_12))
    spikebins = TBF_Estimate(VLMN_12,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_13))
    spikebins = TBF_Estimate(VLMN_13,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_14))
    spikebins = TBF_Estimate(VLMN_14,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))

def PlotHist_CalculateSwimFreq(Title):

    nrns = np.arange(0,len(VLMN_15))
    spikebins = TBF_Estimate(VLMN_15,nrns,0,1000)
    spikebins = np.concatenate(spikebins,axis=0)

    return spikebins

spikebins = PlotHist_CalculateSwimFreq(Title)
spikebins.sort
idx = (spikebins>35)*(spikebins<70)
print(np.mean(spikebins[np.where(idx)]))
np.std(spikebins[np.where(idx)])
print(1000/np.mean(spikebins[np.where(idx)]))