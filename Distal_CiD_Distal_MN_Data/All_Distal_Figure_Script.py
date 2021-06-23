

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

VLIC_1 = np.load('VLIC_run1_all_distal.npy') 
VLCiD_1 = np.load('VLCiD_run1_all_distal.npy')
VLCiA_1 = np.load('VLCiA_run1_all_distal.npy')

VLIC_2 = np.load('VLIC_run2_all_distal.npy') 
VLCiD_2 = np.load('VLCiD_run2_all_distal.npy')
VLCiA_2 = np.load('VLCiA_run2_all_distal.npy')

VLIC_3 = np.load('VLIC_run3_all_distal.npy') 
VLCiD_3 = np.load('VLCiD_run3_all_distal.npy')
VLCiA_3 = np.load('VLCiA_run3_all_distal.npy')

VLIC_4 = np.load('VLIC_run4_all_distal.npy') 
VLCiD_4 = np.load('VLCiD_run4_all_distal.npy')
VLCiA_4 = np.load('VLCiA_run4_all_distal.npy')

VLIC_5 = np.load('VLIC_run5_all_distal.npy') 
VLCiD_5 = np.load('VLCiD_run5_all_distal.npy')
VLCiA_5 = np.load('VLCiA_run5_all_distal.npy')

VLIC_6 = np.load('VLIC_run6_all_distal.npy') 
VLCiD_6 = np.load('VLCiD_run6_all_distal.npy')
VLCiA_6 = np.load('VLCiA_run6_all_distal.npy')

VLIC_7 = np.load('VLIC_run7_all_distal.npy') 
VLCiD_7 = np.load('VLCiD_run7_all_distal.npy')
VLCiA_7 = np.load('VLCiA_run7_all_distal.npy')

VLIC_8 = np.load('VLIC_run8_all_distal.npy') 
VLCiD_8 = np.load('VLCiD_run8_all_distal.npy')
VLCiA_8 = np.load('VLCiA_run8_all_distal.npy')

VLIC_9 = np.load('VLIC_run9_all_distal.npy') 
VLCiD_9 = np.load('VLCiD_run9_all_distal.npy')
VLCiA_9 = np.load('VLCiA_run9_all_distal.npy')

VLIC_10 = np.load('VLIC_run10_all_distal.npy') 
VLCiD_10 = np.load('VLCiD_run10_all_distal.npy')
VLCiA_10 = np.load('VLCiA_run10_all_distal.npy')

VLIC_11 = np.load('VLIC_run11_all_distal.npy') 
VLCiD_11 = np.load('VLCiD_run11_all_distal.npy')
VLCiA_11 = np.load('VLCiA_run11_all_distal.npy')

VLIC_12 = np.load('VLIC_run12_all_distal.npy') 
VLCiD_12 = np.load('VLCiD_run12_all_distal.npy')
VLCiA_12 = np.load('VLCiA_run12_all_distal.npy')

VLIC_13 = np.load('VLIC_run13_all_distal.npy') 
VLCiD_13 = np.load('VLCiD_run13_all_distal.npy')
VLCiA_13 = np.load('VLCiA_run13_all_distal.npy')

VLIC_14 = np.load('VLIC_run14_all_distal.npy') 
VLCiD_14 = np.load('VLCiD_run14_all_distal.npy')
VLCiA_14 = np.load('VLCiA_run14_all_distal.npy')

VLIC_15 = np.load('VLIC_run15_all_distal.npy') 
VLCiD_15 = np.load('VLCiD_run15_all_distal.npy')
VLCiA_15 = np.load('VLCiA_run15_all_distal.npy')

Time = np.load('Time_run1_all_distal.npy')


Connectome_Weights = np.load('CW_run1_all_distal.npy',allow_pickle=True)
VLMN = np.concatenate((VLMN_1,VLMN_2,VLMN_3,VLMN_4,VLMN_5,VLMN_6,VLMN_7,VLMN_8,VLMN_9,VLMN_10,VLMN_11,VLMN_12,VLMN_13,VLMN_14,VLMN_15))


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

    fig, ax = plt.subplots()

    nrns = np.arange(0,len(VLMN))
    spikebins = TBF_Estimate(VLMN,nrns,0,1000)

    plt.ion()
    spikebins = np.concatenate(spikebins,axis=0)

    #for x in np.arange(len(spikebins)):

    bins = np.linspace(0,100,50)
    #n, bins, patches = plt.hist(spikebins[x],bins,alpha=0.8,color='black')
    sns.distplot(spikebins,bins,kde=False,hist=True,color='#4f2509',hist_kws={'alpha':1},kde_kws={'bw':1.5}) #,kde_kws={'bw':5}
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    #plt.style.use('seaborn-white')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis="x", direction="out", length=10, width=1, color="black",labelbottom=False)
    ax.tick_params(axis="y", direction="out", length=10, width=1, color="black",labelleft=False)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="y", which="minor",direction="out", length=5, width=1, color="black")
    ax.set_xlim([0,100])
    ax.set_ylim([0,3000])
    #ax.xaxis.set_minor_locator(AutoMinorLocator())

    plt.show()

    return spikebins

# def Plot_polar(Title):

#     nrns = [12,13,14]

#     spikes_indL = []
#     spikes_indL_hist = []

#     for counter,neuron in enumerate(nrns):

#         thres = 0
#         spikest_L = np.where(VLMN_1[neuron]>thres) # find all times when neuron goes above threshold 
#         peaks_L = VLMN_1[neuron][spikest_L] # the peak values for each neuron at the times found in the step above 
#         spikes_indt_L = spikest_L[0]/10 # find spikes times and convert to milliseconds 

#         spikes_indL.append(spikes_indt_L) # setting up array for event/raster plot below organized by neuron
#         spikes_indL_hist.extend(spikes_indt_L) # setting up array with neuron identity thrown away (only spikes times)

#         #ax[0].plot(Time,VLMN[neuron]-(counter*100), color='green') # plot all neuron traces with shifts down for each subsequent neuron 

#         #ax[0].plot(spikes_indt_L,peaks_L-(counter*100),'*',color='dimgray') # plot peak positions to verify that spikes are being detected accurately 
#         #ax[0].text(1030,-30,"Left MN",color='green')

#         #ax[0].set_xlim([0,1000])

#     #ax[0].set_title(Title)

#     #ax[1].eventplot(spikes_indL,color='green') # create raster plot 
#     #ax[1].eventplot(spikes_indR,color='black')
#     #ax[1].set_xlim([0,1000])

#     bins=np.linspace(0,1000,200) # 5 ms width bins (i.e. 1000/200)
#     countsL, binsL = np.histogram(spikes_indL_hist,bins) # find frequency of spikes within the bins above
#     #countsR, binsR = np.histogram(spikes_indR_hist,bins)
#     #plt.figure()
#     #plt.hist(spikes_indL_hist,bins,color='green',alpha=0.7,density=False) # plot histogram 
#     #plt.xlim([0,1000])

#     peaksL, _ = find_peaks(countsL, height=2) # height is important here (basically the prominence) the minimum height of peak for it to be detected (at least 2 spikes within bin)
#     #peaksR, _ = find_peaks(countsR, height=2)
#     #plt.plot(binsL[peaksL], countsL[peaksL], "v",color='red') # plot detected peaks in histogram
#     #ax[2].plot(binsR[peaksR], countsR[peaksR], "v",color='red')

#     polarspikes_Left_all = []
#     interval_all = []
#     interval_temp = []

#     nrns = np.arange(0,len(VLMN_1))

#     spikes_indL = []
#     spikes_indL_hist = []

#     for counter,neuron in enumerate(nrns):

#         thres = 0
#         spikest_L = np.where(VLMN_1[neuron]>thres) # find all times when neuron goes above threshold 
#         peaks_L = VLMN_1[neuron][spikest_L] # the peak values for each neuron at the times found in the step above 
#         spikes_indt_L = spikest_L[0]/10 # find spikes times and convert to milliseconds 

#         spikes_indL.append(spikes_indt_L) # setting up array for event/raster plot below organized by neuron
#         spikes_indL_hist.extend(spikes_indt_L) # setting up array with neuron identity thrown away (only spikes times)

#     for x,y in zip(binsL[peaksL],binsL[peaksL[1:]]):

#         interval = y-x # finding the time interval between peaks detected in the histogram 

#         if interval > 90: # change this interval based on the distribution of the interval histogram 
#             continue

#         if interval < 20: 
#             continue

#         indext_L = np.where(np.logical_and(spikes_indt_L>x, spikes_indt_L<y)) # find the index of the spikes that occurred between the two peaks

#         polarspikes_Left = []

#         for spike in spikes_indt_L[indext_L]: # iterate this loop for every spike that occurred between two peaks

#             normspike = spike - x # normalize spike times to the start of the interval (first peak)
#             #print('normspike: ' + str(normspike))

#             polarspike = (2*np.pi*normspike)/interval # convert peak time to polar coordinates normalized to the interval length 

#             #print('polarspike: ' +str(polarspike))

#             polarspikes_Left.append(polarspike)

#         polarspikes_Left_all.extend(polarspikes_Left)


#     sin_angle_all_left = []
#     cos_angle_all_left = []

#     sin_angle_temp_left = []
#     cos_angle_temp_left = []

#     for radian in polarspikes_Left_all:
#         #print(radian)
#         #angle = radian*(180/pi)
#         sin_angle = np.sin(radian)
#         cos_angle = np.cos(radian)

#         sin_angle_temp_left.append(sin_angle)
#         cos_angle_temp_left.append(cos_angle)

#     sin_angle_all_left.extend(sin_angle_temp_left)
#     cos_angle_all_left.extend(cos_angle_temp_left)

#     X_left = np.mean(sin_angle_all_left)
#     Y_left = np.mean(cos_angle_all_left)

#     #print(X_left)
#     #print(Y_left)

#     vectorlen_left = np.sqrt((X_left**2) + (Y_left**2))

#     fig2,ax2 = plt.subplots(1,1,subplot_kw=dict(projection='polar'))
#     ax2.set_xticklabels(['0', '', '', '', '0.5', '', '', ''])
#     ax2.set_yticklabels(['','','','',''])
#     ax2.set_yticks([])
#     #ax2.set_xlim([0,360])
#     ax2.set_ylim([0,1.1])

#     radiusL = np.ones(len(polarspikes_Left_all))

#     ax2.plot(polarspikes_Left_all,radiusL,'o',color='#000000',alpha=0.7)
#     #ax2.plot(polarspikes_Right_all,radiusR,'o',color='black',alpha=0.7)
#     ax2.quiver(0,0,np.mean(polarspikes_Left_all),vectorlen_left, color='#000000', angles="xy", scale_units='xy', scale=1.)
#     #ax2.quiver(0,0,mean(polarspikes_Right_all),vectorlen_right, color='black', angles="xy", scale_units='xy', scale=1.)
#     #ax2.text(2*np.pi,1.25,"Left MN",color='green')
#     #ax2.text(1.95*np.pi,1.25,'Left Vector Length: ' + str(round(vectorlen_left,3)),color='green')
#     ax2.set_title(Title + ' neurons: ' + str(nrns))

#     ##########################################################

#     nrns = [12,13,14]

#     spikes_indL = []
#     spikes_indL_hist = []

#     for counter,neuron in enumerate(nrns):

#         thres = 0
#         spikest_L = np.where(VLMN_2[neuron]>thres) # find all times when neuron goes above threshold 
#         peaks_L = VLMN_2[neuron][spikest_L] # the peak values for each neuron at the times found in the step above 
#         spikes_indt_L = spikest_L[0]/10 # find spikes times and convert to milliseconds 

#         spikes_indL.append(spikes_indt_L) # setting up array for event/raster plot below organized by neuron
#         spikes_indL_hist.extend(spikes_indt_L) # setting up array with neuron identity thrown away (only spikes times)

#         #ax[0].plot(Time,VLMN[neuron]-(counter*100), color='green') # plot all neuron traces with shifts down for each subsequent neuron 

#         #ax[0].plot(spikes_indt_L,peaks_L-(counter*100),'*',color='dimgray') # plot peak positions to verify that spikes are being detected accurately 
#         #ax[0].text(1030,-30,"Left MN",color='green')

#         #ax[0].set_xlim([0,1000])

#     #ax[0].set_title(Title)

#     #ax[1].eventplot(spikes_indL,color='green') # create raster plot 
#     #ax[1].eventplot(spikes_indR,color='black')
#     #ax[1].set_xlim([0,1000])

#     bins=np.linspace(0,1000,200) # 5 ms width bins (i.e. 1000/200)
#     countsL, binsL = np.histogram(spikes_indL_hist,bins) # find frequency of spikes within the bins above
#     #countsR, binsR = np.histogram(spikes_indR_hist,bins)
#     #plt.figure()
#     #plt.hist(spikes_indL_hist,bins,color='blue',alpha=0.7,density=False) # plot histogram 
#     #plt.xlim([0,1000])

#     peaksL, _ = find_peaks(countsL, height=2) # height is important here (basically the prominence) the minimum height of peak for it to be detected (at least 2 spikes within bin)
#     #peaksR, _ = find_peaks(countsR, height=2)
#     #plt.plot(binsL[peaksL], countsL[peaksL], "v",color='red') # plot detected peaks in histogram
#     #ax[2].plot(binsR[peaksR], countsR[peaksR], "v",color='red')

#     polarspikes_Left_all = []
#     interval_all = []
#     interval_temp = []

#     nrns = np.arange(0,len(VLMN_2))

#     spikes_indL = []
#     spikes_indL_hist = []

#     for counter,neuron in enumerate(nrns):

#         thres = 0
#         spikest_L = np.where(VLMN_2[neuron]>thres) # find all times when neuron goes above threshold 
#         peaks_L = VLMN_2[neuron][spikest_L] # the peak values for each neuron at the times found in the step above 
#         spikes_indt_L = spikest_L[0]/10 # find spikes times and convert to milliseconds 

#         spikes_indL.append(spikes_indt_L) # setting up array for event/raster plot below organized by neuron
#         spikes_indL_hist.extend(spikes_indt_L) # setting up array with neuron identity thrown away (only spikes times)

#     for x,y in zip(binsL[peaksL],binsL[peaksL[1:]]):

#         interval = y-x # finding the time interval between peaks detected in the histogram 

#         if interval > 90: # change this interval based on the distribution of the interval histogram 
#             continue

#         if interval < 20: 
#             continue

#         indext_L = np.where(np.logical_and(spikes_indt_L>x, spikes_indt_L<y)) # find the index of the spikes that occurred between the two peaks

#         polarspikes_Left = []

#         for spike in spikes_indt_L[indext_L]: # iterate this loop for every spike that occurred between two peaks

#             normspike = spike - x # normalize spike times to the start of the interval (first peak)
#             #print('normspike: ' + str(normspike))

#             polarspike = (2*np.pi*normspike)/interval # convert peak time to polar coordinates normalized to the interval length 

#             #print('polarspike: ' +str(polarspike))

#             polarspikes_Left.append(polarspike)

#         polarspikes_Left_all.extend(polarspikes_Left)


#     sin_angle_all_left = []
#     cos_angle_all_left = []

#     sin_angle_temp_left = []
#     cos_angle_temp_left = []

#     for radian in polarspikes_Left_all:
#         #print(radian)
#         #angle = radian*(180/pi)
#         sin_angle = np.sin(radian)
#         cos_angle = np.cos(radian)

#         sin_angle_temp_left.append(sin_angle)
#         cos_angle_temp_left.append(cos_angle)

#     sin_angle_all_left.extend(sin_angle_temp_left)
#     cos_angle_all_left.extend(cos_angle_temp_left)

#     X_left = np.mean(sin_angle_all_left)
#     Y_left = np.mean(cos_angle_all_left)

#     #print(X_left)
#     #print(Y_left)

#     vectorlen_left = np.sqrt((X_left**2) + (Y_left**2))

#     #fig2,ax2 = plt.subplots(1,1,subplot_kw=dict(projection='polar'))
#     #ax2.set_xticklabels(['0', '', '', '', '0.5', '', '', ''])
#     #ax2.set_yticklabels(['','','','',''])
#     #ax2.set_yticks([])
#     #ax2.set_xlim([0,360])
#     #ax2.set_ylim([0,1.1])

#     radiusL = np.ones(len(polarspikes_Left_all))

#     ax2.plot(polarspikes_Left_all,radiusL,'o',color='#737373',alpha=0.7)
#     #ax2.plot(polarspikes_Right_all,radiusR,'o',color='black',alpha=0.7)
#     ax2.quiver(0,0,np.mean(polarspikes_Left_all),vectorlen_left, color='#737373', angles="xy", scale_units='xy', scale=1.)
#     #ax2.quiver(0,0,mean(polarspikes_Right_all),vectorlen_right, color='black', angles="xy", scale_units='xy', scale=1.)
#     #ax2.text(2*np.pi,1.25,"Left MN",color='blue')
#     #ax2.text(1.95*np.pi,1.25,'Left Vector Length: ' + str(round(vectorlen_left,3)),color='blue')
#     ax2.set_title(Title + ' neurons: ' + str(nrns))

#     ##########################################################

#     nrns = [12,13,14]

#     spikes_indL = []
#     spikes_indL_hist = []

#     for counter,neuron in enumerate(nrns):

#         thres = 0
#         spikest_L = np.where(VLMN_3[neuron]>thres) # find all times when neuron goes above threshold 
#         peaks_L = VLMN_3[neuron][spikest_L] # the peak values for each neuron at the times found in the step above 
#         spikes_indt_L = spikest_L[0]/10 # find spikes times and convert to milliseconds 

#         spikes_indL.append(spikes_indt_L) # setting up array for event/raster plot below organized by neuron
#         spikes_indL_hist.extend(spikes_indt_L) # setting up array with neuron identity thrown away (only spikes times)

#         #ax[0].plot(Time,VLMN[neuron]-(counter*100), color='green') # plot all neuron traces with shifts down for each subsequent neuron 

#         #ax[0].plot(spikes_indt_L,peaks_L-(counter*100),'*',color='dimgray') # plot peak positions to verify that spikes are being detected accurately 
#         #ax[0].text(1030,-30,"Left MN",color='green')

#         #ax[0].set_xlim([0,1000])

#     #ax[0].set_title(Title)

#     #ax[1].eventplot(spikes_indL,color='green') # create raster plot 
#     #ax[1].eventplot(spikes_indR,color='black')
#     #ax[1].set_xlim([0,1000])

#     bins=np.linspace(0,1000,200) # 5 ms width bins (i.e. 1000/200)
#     countsL, binsL = np.histogram(spikes_indL_hist,bins) # find frequency of spikes within the bins above
#     #countsR, binsR = np.histogram(spikes_indR_hist,bins)
#     #plt.figure()
#     #plt.hist(spikes_indL_hist,bins,color='magenta',alpha=0.7,density=False) # plot histogram 
#     #plt.xlim([0,1000])

#     peaksL, _ = find_peaks(countsL, height=2) # height is important here (basically the prominence) the minimum height of peak for it to be detected (at least 2 spikes within bin)
#     #peaksR, _ = find_peaks(countsR, height=2)
#     #plt.plot(binsL[peaksL], countsL[peaksL], "v",color='red') # plot detected peaks in histogram
#     #ax[2].plot(binsR[peaksR], countsR[peaksR], "v",color='red')

#     polarspikes_Left_all = []
#     interval_all = []
#     interval_temp = []

#     nrns = np.arange(0,len(VLMN_3))

#     spikes_indL = []
#     spikes_indL_hist = []

#     for counter,neuron in enumerate(nrns):

#         thres = 0
#         spikest_L = np.where(VLMN_3[neuron]>thres) # find all times when neuron goes above threshold 
#         peaks_L = VLMN_3[neuron][spikest_L] # the peak values for each neuron at the times found in the step above 
#         spikes_indt_L = spikest_L[0]/10 # find spikes times and convert to milliseconds 

#         spikes_indL.append(spikes_indt_L) # setting up array for event/raster plot below organized by neuron
#         spikes_indL_hist.extend(spikes_indt_L) # setting up array with neuron identity thrown away (only spikes times)

#     for x,y in zip(binsL[peaksL],binsL[peaksL[1:]]):

#         interval = y-x # finding the time interval between peaks detected in the histogram 

#         if interval > 90: # change this interval based on the distribution of the interval histogram 
#             continue

#         if interval < 20: 
#             continue

#         indext_L = np.where(np.logical_and(spikes_indt_L>x, spikes_indt_L<y)) # find the index of the spikes that occurred between the two peaks

#         polarspikes_Left = []

#         for spike in spikes_indt_L[indext_L]: # iterate this loop for every spike that occurred between two peaks

#             normspike = spike - x # normalize spike times to the start of the interval (first peak)
#             #print('normspike: ' + str(normspike))

#             polarspike = (2*np.pi*normspike)/interval # convert peak time to polar coordinates normalized to the interval length 

#             #print('polarspike: ' +str(polarspike))

#             polarspikes_Left.append(polarspike)

#         polarspikes_Left_all.extend(polarspikes_Left)


#     sin_angle_all_left = []
#     cos_angle_all_left = []

#     sin_angle_temp_left = []
#     cos_angle_temp_left = []

#     for radian in polarspikes_Left_all:
#         #print(radian)
#         #angle = radian*(180/pi)
#         sin_angle = np.sin(radian)
#         cos_angle = np.cos(radian)

#         sin_angle_temp_left.append(sin_angle)
#         cos_angle_temp_left.append(cos_angle)

#     sin_angle_all_left.extend(sin_angle_temp_left)
#     cos_angle_all_left.extend(cos_angle_temp_left)

#     X_left = np.mean(sin_angle_all_left)
#     Y_left = np.mean(cos_angle_all_left)

#     #print(X_left)
#     #print(Y_left)

#     vectorlen_left = np.sqrt((X_left**2) + (Y_left**2))

#     #fig2,ax2 = plt.subplots(1,1,subplot_kw=dict(projection='polar'))
#     #ax2.set_xticklabels(['0', '', '', '', '0.5', '', '', ''])
#     #ax2.set_yticklabels(['','','','',''])
#     #ax2.set_yticks([])
#     #ax2.set_xlim([0,360])
#     #ax2.set_ylim([0,1.1])

#     radiusL = np.ones(len(polarspikes_Left_all))

#     plt.style.use('seaborn-white')


#     ax2.plot(polarspikes_Left_all,radiusL,'o',color='#737373',alpha=0.7)
#     #ax2.plot(polarspikes_Right_all,radiusR,'o',color='black',alpha=0.7)
#     ax2.quiver(0,0,np.mean(polarspikes_Left_all),vectorlen_left, color='#737373', angles="xy", scale_units='xy', scale=1.)
#     #ax2.quiver(0,0,mean(polarspikes_Right_all),vectorlen_right, color='black', angles="xy", scale_units='xy', scale=1.)
#     #ax2.text(2*np.pi,1.25,"Left MN",color='magenta')
#     #ax2.text(1.95*np.pi,1.25,'Left Vector Length: ' + str(round(vectorlen_left,3)),color='magenta')
#     ax2.set_title(Title)

def ChemicalSynapseWeightMatrix(Connectome_Weights,Title):
    
    nIC = 5
    nMN = 15
    nCiD = 12
    nCiA =10
    
    Connectome_Weights[()]['LW_IC_IC']=np.zeros((nIC,nIC))
    Connectome_Weights[()]['LW_IC_MN'] = np.zeros((nIC,nMN))
    Connectome_Weights[()]['LW_IC_CiD']=np.zeros((nIC,nCiD))
    Connectome_Weights[()]['LW_IC_CiA']=np.zeros((nIC,nCiA))

    Connectome_Weights[()]['LW_MN_MN']=np.zeros((nMN,nMN))
    Connectome_Weights[()]['LW_MN_CiD']=np.zeros((nMN,nCiD))
    Connectome_Weights[()]['LW_MN_CiA'] = np.zeros((nMN, nCiA))

    Connectome_Weights[()]['LW_CiA_CiA']=np.zeros((nCiA,nCiA))


    
    stack = np.hstack((Connectome_Weights[()]['LGap_IC_IC'],Connectome_Weights[()]['LW_IC_MN'],Connectome_Weights[()]['LGap_IC_CiD'],Connectome_Weights[()]['LW_IC_CiA']))

    stack1=np.hstack((Connectome_Weights[()]['LW_IC_MN'].T,Connectome_Weights[()]['LW_MN_MN'],Connectome_Weights[()]['LW_MN_CiD'],Connectome_Weights[()]['LW_MN_CiA']))

    stack2 = np.hstack((Connectome_Weights[()]['LGap_IC_CiD'].T,Connectome_Weights[()]['LW_CiD_MN'],Connectome_Weights[()]['LW_CiD_CiD'],Connectome_Weights[()]['LW_CiD_CiA']))

    stack3 = np.hstack((Connectome_Weights[()]['LW_CiA_IC'],Connectome_Weights[()]['LW_CiA_MN'],Connectome_Weights[()]['LW_CiA_CiD'],Connectome_Weights[()]['LW_CiA_CiA']))

    stackf = np.vstack((stack,stack1,stack2,stack3)).T

    plt.ion()

    fig,ax =plt.subplots()
    #im = ax.imshow(stackf)

    #ax.set_title(str(Title))
    #ax.set_xlabel('Presynaptic Neuron')
    ax.set_xticks([2,12.5,26,37])
    #ax.set_xticklabels(['IC','MN','CiD','CiA'])
    #ax.set_ylabel('Postsynaptic Neuron')
    ax.set_yticks([2,12.5,26,37])
    #ax.set_yticklabels(['IC','MN','CiD','CiA'])
    #ax.xaxis.tick_top()

    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    ax.tick_params(axis="x", direction="out", length=10, width=1, color="black",labelbottom=False)
    ax.tick_params(axis="y", direction="out", length=10, width=1, color="black",labelleft=False)
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    #ax.tick_params(axis="y", which="minor",direction="out", length=5, width=1, color="black")
    #ax.set_xlim([0,400])
    #ax.set_ylim([0,37])


    xcoords = [4.5,19.5,31.5]
    for xc in xcoords:
        plt.axvline(x=xc,color='w',linewidth=0.5)
        
    ycoords = [4.5,19.5,31.5]
    for yc in ycoords:
        plt.axhline(y=yc,color='w',linewidth=0.5)
    
    #plt.imshow(stackf,cmap=plt.cm.BuPu_r,norm=clr.SymLogNorm(linthresh=0.001,vmin=0, vmax=5.0))
    #plt.style.use('seaborn-white')
    plt.imshow(stackf,cmap=plt.cm.viridis,vmin=0,vmax=0.5)
    #plt.imshow(stackf,cmap=plt.cm.BuPu_r,norm=clr.SymLogNorm(linthresh=0.05,vmin=0, vmax=0.5))
    #plt.colorbar()
    plt.show()

def Rasterplot_beatnglide(Time, neuron_list,Title):

    plt.style.use('seaborn-ticks')
    fig, ax = plt.subplots(1,1)
    spikes = []
    spikes_ind = []
    colorsf = []

    colorsf = []

    #colors1 = ['#9D95AD'.format(i) for i in np.arange(len(neuron_list[0]))]
    #colorsf.extend(colors1)
    colors0 = ['#9F94AF'.format(i) for i in np.arange(len(neuron_list[0]))]
    colorsf.extend(colors0)
    colors1 = ['#572200'.format(i) for i in np.arange(len(neuron_list[1]))]
    colorsf.extend(colors1)
    colors2 = ['#377E21'.format(i) for i in np.arange(len(neuron_list[2]))]
    colorsf.extend(colors2)
    colors3 = ['#000000'.format(i) for i in np.arange(len(neuron_list[3]))]
    colorsf.extend(colors3)


    plt.ion()

    for population in np.arange(len(neuron_list)):
        neuron_type = neuron_list[population]
        alt = str(neuron_type)
        
        for neuron in np.arange(len(neuron_type)):
            if alt == str(VLIC_4):
                thres = -10
            else:
                thres = 0
            #thres = 0
            spikest=np.where(neuron_type[neuron,:]>thres)
            spikes_indt = spikest[0]/10
        
            spikes.append(spikest[0])
            spikes_ind.append(spikes_indt)

    #colors1 = ['C{}'.format(i) for i in range(6)]
    ax.eventplot(spikes_ind,colors=colorsf,linewidth=2.5)
    ax.set_xlim([0,1000])
    
    #ax.set_title(str(Title))
    
    #ax.text(1020,3,r"VLIC",fontsize = 10,zorder = 5,color = '#9D95AD',fontweight = 'bold')
    #ax.text(1020,12,r"VLMN",fontsize = 10,zorder = 5,color = '#4F2509',fontweight = 'bold')
    #ax.text(1020,25,r"VLCiD",fontsize = 10,zorder = 5,color = '#377E21',fontweight = 'bold')
    #ax.text(1020,35,r"VLCiA",fontsize = 10,zorder = 5,color = '#000000',fontweight = 'bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis="x", direction="out", length=10, width=1, color="black",labelbottom=False)
    ax.tick_params(axis="y", direction="out", length=10, width=1, color="white",labelleft=False)
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    #ax.tick_params(axis="y", which="minor",direction="out", length=5, width=1, color="black")
    ax.set_xlim([0,400])
    ax.set_ylim([0,42])

    spikes_indL=[]

    # for neuron in np.arange(len(neurontype_left)):
    #     thres=0
    #     spikest=np.where(neurontype_left[neuron,:]>thres)
    #     spikes_indt = spikest[0]/10
    
    #     spikes_indL.append(spikes_indt)
    
    plt.show()

Title = 'Tonic Drive Distal CiA_CiD Distal CiA_MN 3 Runs'
neuron_list = [VLIC_4,VLMN_4,VLCiD_4,VLCiA_4] 
Rasterplot_beatnglide(Time,neuron_list,Title)
spikebins = PlotHist_CalculateSwimFreq(Title)
#Plot_polar(Title)
ChemicalSynapseWeightMatrix(Connectome_Weights, Title)