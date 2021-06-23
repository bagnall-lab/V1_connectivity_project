import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np

#from Beat_and_glide_Ipsilateral_Only_bad import *
from Beat_and_glide_Ipsilateral_Only_Final import *

((VLIC), (VLMN), (VLCiD),
(VLCiA),Time, Connectome_Weights, sum_CIA_MN, sum_CIA_CiD) = connectome_beat_glide_ipsilateral(rand=0, stim0=0.35, sigma=0.0,
                                                     tmax=1000, dt=0.1, E_glu=0, E_gly=-70, c=0.55,
                                                     nIC=5, nMN=15, nCiD=12, nCiA=10, 
                                                     R_str=1.0)

def Rasterplot_beatnglide(Time, neuron_list,neurontype_left,Title):

    fig, ax = plt.subplots(1,1)
    spikes = []
    spikes_ind = []
    colorsf = []

    colorsf = []

    colors1 = ['C0'.format(i) for i in np.arange(len(neuron_list[0]))]
    colorsf.extend(colors1)
    colors2 = ['C1'.format(i) for i in np.arange(len(neuron_list[1]))]
    colorsf.extend(colors2)
    colors3 = ['C2'.format(i) for i in np.arange(len(neuron_list[2]))]
    colorsf.extend(colors3)
    colors4 = ['C3'.format(i) for i in np.arange(len(neuron_list[3]))]
    colorsf.extend(colors4)


    plt.ion()

    for population in np.arange(len(neuron_list)):
        neuron_type = neuron_list[population]
        alt = str(neuron_type)
        
        for neuron in np.arange(len(neuron_type)):
            if alt == str(VLIC):
                thres = -10
            else:
                thres = 0
            spikest=np.where(neuron_type[neuron,:]>thres)
            spikes_indt = spikest[0]/10
        
            spikes.append(spikest[0])
            spikes_ind.append(spikes_indt)

    #colors1 = ['C{}'.format(i) for i in range(6)]
    ax.eventplot(spikes_ind,colors=colorsf)
    ax.set_xlim([0,1000])
    
    ax.set_title(str(Title))
    
    ax.text(1020,3,r"VLIC",fontsize = 10,zorder = 5,color = 'C0',fontweight = 'bold')
    ax.text(1020,12,r"VLMN",fontsize = 10,zorder = 5,color = 'C1',fontweight = 'bold')
    ax.text(1020,25,r"VLCiD",fontsize = 10,zorder = 5,color = 'C2',fontweight = 'bold')
    ax.text(1020,35,r"VLCiA",fontsize = 10,zorder = 5,color = 'C3',fontweight = 'bold')

    spikes_indL=[]

    for neuron in np.arange(len(neurontype_left)):
        thres=0
        spikest=np.where(neurontype_left[neuron,:]>thres)
        spikes_indt = spikest[0]/10
    
        spikes_indL.append(spikes_indt)
    
    plt.show()

def Show_peaks(Title):

    nrns = [5,6,7,8,9]
    fig, ax = plt.subplots(3,1,gridspec_kw={'height_ratios': [4, 2, 2]})

    spikes_indL = []
    spikes_indL_hist = []
    spikes_indR = []
    spikes_indR_hist = []

    for counter,neuron in enumerate(nrns):

        thres = 0
        spikest_L = np.where(VLMN[neuron]>thres) # find all times when neuron goes above threshold 
        peaks_L = VLMN[neuron][spikest_L] # the peak values for each neuron at the times found in the step above 
        spikes_indt_L = spikest_L[0]/10 # find spikes times and convert to milliseconds 

        spikes_indL.append(spikes_indt_L) # setting up array for event/raster plot below organized by neuron
        spikes_indL_hist.extend(spikes_indt_L) # setting up array with neuron identity thrown away (only spikes times)

        ax[0].plot(Time,VLMN[neuron]-(counter*100), color='green') # plot all neuron traces with shifts down for each subsequent neuron 
        ax[0].plot(spikes_indt_L,peaks_L-(counter*100),'*',color='dimgray') # plot peak positions to verify that spikes are being detected accurately 
        ax[0].text(1030,-30,"Left MN",color='green')
        ax[0].set_xlim([0,1000])

    ax[0].set_title(Title)

    ax[1].eventplot(spikes_indL,color='green') # create raster plot 
    ax[1].set_xlim([0,1000])

    bins=np.linspace(0,1000,200) # 5 ms width bins (i.e. 1000/200)
    countsL, binsL = np.histogram(spikes_indL_hist,bins) # find frequency of spikes within the bins above
    ax[2].hist(spikes_indL_hist,bins,color='green',alpha=0.7,density=False) # plot histogram 
    ax[2].set_xlim([0,1000])

    peaksL, _ = find_peaks(countsL, height=2) # height is important here (basically the prominence) the minimum height of peak for it to be detected (at least 2 spikes within bin)
    ax[2].plot(binsL[peaksL], countsL[peaksL], "v",color='red') # plot detected peaks in histogram

def ChemicalSynapseWeightMatrix(Connectome_Weights,Title):
    
    nIC = 5
    nMN = 15
    nCiD = 12
    nCiA =10
    
    Connectome_Weights['LW_IC_IC']=zeros((nIC,nIC))
    Connectome_Weights['LW_IC_MN'] = zeros((nIC,nMN))
    Connectome_Weights['LW_IC_CiD']=zeros((nIC,nCiD))
    Connectome_Weights['LW_IC_CiA']=zeros((nIC,nCiA))

    Connectome_Weights['LW_MN_MN']=zeros((nMN,nMN))
    Connectome_Weights['LW_MN_CiD']=zeros((nMN,nCiD))
    Connectome_Weights['LW_MN_CiA'] = zeros((nMN, nCiA))

    Connectome_Weights['LW_CiA_CiA']=zeros((nCiA,nCiA))


    
    stack = np.hstack((Connectome_Weights['LW_IC_IC'],Connectome_Weights['LW_IC_MN'],Connectome_Weights['LW_IC_CiD'],Connectome_Weights['LW_IC_CiA']))

    stack1=np.hstack((Connectome_Weights['LW_IC_MN'].T,Connectome_Weights['LW_MN_MN'],Connectome_Weights['LW_MN_CiD'],Connectome_Weights['LW_MN_CiA']))

    stack2 = np.hstack((Connectome_Weights['LW_IC_CiD'].T,Connectome_Weights['LW_CiD_MN'],Connectome_Weights['LW_CiD_CiD'],Connectome_Weights['LW_CiD_CiA']))

    stack3 = np.hstack((Connectome_Weights['LW_CiA_IC'],Connectome_Weights['LW_CiA_MN'],Connectome_Weights['LW_CiA_CiD'],Connectome_Weights['LW_CiA_CiA']))

    stackf = np.vstack((stack,stack1,stack2,stack3)).T

    plt.ion()

    fig,ax =plt.subplots()
    #im = ax.imshow(stackf)

    ax.set_title(str(Title))
    ax.set_xlabel('Presynaptic Neuron')
    ax.set_xticks([2.5,12.5,27.5,40.5])
    ax.set_xticklabels(['IC','MN','CiD','CiA'])
    ax.set_ylabel('Postsynaptic Neuron')
    ax.set_yticks([2.5,12.5,27.5,40.5])
    ax.set_yticklabels(['IC','MN','CiD','CiA'])
    ax.xaxis.tick_top()
    xcoords = [4.5,19.5,31.5]
    for xc in xcoords:
        plt.axvline(x=xc,color='w',linewidth=0.5)
        
    ycoords = [4.5,19.5,31.5]
    for yc in ycoords:
        plt.axhline(y=yc,color='w',linewidth=0.5)
    
    #plt.imshow(stackf,cmap=plt.cm.BuPu_r,norm=clr.SymLogNorm(linthresh=0.001,vmin=0, vmax=5.0))
    plt.imshow(stackf,cmap=plt.cm.BuPu_r)
    plt.colorbar()
    plt.show()
    
    
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
        
        spikebins.append(diff(spikes_ind[nrn][indext]))
        
    return spikebins


def PlotHist_CalculateSwimFreq(Title):

    plt.figure()

    nrns = [6,7,8,9,10,11,12,13,14]
    spikebins = TBF_Estimate(VLMN,nrns,0,400)

    plt.ion()

    for x in np.arange(len(spikebins)):

        bins = np.linspace(0,100,50)
        #n, bins, patches = plt.hist(spikebins[x],bins,alpha=0.8,color='black')
        sns.distplot(spikebins[x],bins,kde=False,hist=True,color='green',kde_kws={'bw':1.5}) #,kde_kws={'bw':5}
        plt.xlabel('Time(ms)')
        plt.ylabel('Frequency')
        plt.title(str(Title))
        plt.style.use('seaborn')


Title = 'Tonic Drive Local CiA_CiD Local CiA_MN 1000ms Simulation'
neuron_list = [VLIC,VLMN,VLCiD,VLCiA] 
Rasterplot_beatnglide(Time,neuron_list,VLMN,Title)
Show_peaks(Title)
ChemicalSynapseWeightMatrix(Connectome_Weights, Title)
PlotHist_CalculateSwimFreq(Title)
print('CIA MN Weights: ' + str(sum_CIA_MN))
print('CIA CiD Weights: ' + str(sum_CIA_CiD))
