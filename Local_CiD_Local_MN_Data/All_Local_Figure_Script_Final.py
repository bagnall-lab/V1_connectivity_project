

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks
import matplotlib.colors as clr
from matplotlib.ticker import AutoMinorLocator


VLMN_1 = np.load('VLMN_run1_all_local.npy')                                                                                                                          
VLMN_2 = np.load('VLMN_run2_all_local.npy')                                                                                                                        
VLMN_3 = np.load('VLMN_run3_all_local.npy')
VLMN_4 = np.load('VLMN_run4_all_local.npy')
VLMN_5 = np.load('VLMN_run5_all_local.npy')
VLMN_6 = np.load('VLMN_run6_all_local.npy')
VLMN_7 = np.load('VLMN_run7_all_local.npy')
VLMN_8 = np.load('VLMN_run8_all_local.npy')
VLMN_9 = np.load('VLMN_run9_all_local.npy')
VLMN_10 = np.load('VLMN_run10_all_local.npy')
VLMN_11 = np.load('VLMN_run11_all_local.npy')
VLMN_12 = np.load('VLMN_run12_all_local.npy')
VLMN_13 = np.load('VLMN_run13_all_local.npy')
VLMN_14 = np.load('VLMN_run14_all_local.npy')
VLMN_15 = np.load('VLMN_run15_all_local.npy')

VLIC_1 = np.load('VLIC_run1_all_local.npy') 
VLCiD_1 = np.load('VLCiD_run1_all_local.npy')
VLCiA_1 = np.load('VLCiA_run1_all_local.npy')

VLIC_2 = np.load('VLIC_run2_all_local.npy') 
VLCiD_2 = np.load('VLCiD_run2_all_local.npy')
VLCiA_2 = np.load('VLCiA_run2_all_local.npy')

VLIC_3 = np.load('VLIC_run3_all_local.npy') 
VLCiD_3 = np.load('VLCiD_run3_all_local.npy')
VLCiA_3 = np.load('VLCiA_run3_all_local.npy')

VLIC_4 = np.load('VLIC_run4_all_local.npy') 
VLCiD_4 = np.load('VLCiD_run4_all_local.npy')
VLCiA_4 = np.load('VLCiA_run4_all_local.npy')

VLIC_5 = np.load('VLIC_run5_all_local.npy') 
VLCiD_5 = np.load('VLCiD_run5_all_local.npy')
VLCiA_5 = np.load('VLCiA_run5_all_local.npy')

VLIC_6 = np.load('VLIC_run6_all_local.npy') 
VLCiD_6 = np.load('VLCiD_run6_all_local.npy')
VLCiA_6 = np.load('VLCiA_run6_all_local.npy')

VLIC_7 = np.load('VLIC_run7_all_local.npy') 
VLCiD_7 = np.load('VLCiD_run7_all_local.npy')
VLCiA_7 = np.load('VLCiA_run7_all_local.npy')

VLIC_8 = np.load('VLIC_run8_all_local.npy') 
VLCiD_8 = np.load('VLCiD_run8_all_local.npy')
VLCiA_8 = np.load('VLCiA_run8_all_local.npy')

VLIC_9 = np.load('VLIC_run9_all_local.npy') 
VLCiD_9 = np.load('VLCiD_run9_all_local.npy')
VLCiA_9 = np.load('VLCiA_run9_all_local.npy')

VLIC_10 = np.load('VLIC_run10_all_local.npy') 
VLCiD_10 = np.load('VLCiD_run10_all_local.npy')
VLCiA_10 = np.load('VLCiA_run10_all_local.npy')

VLIC_11 = np.load('VLIC_run11_all_local.npy') 
VLCiD_11 = np.load('VLCiD_run11_all_local.npy')
VLCiA_11 = np.load('VLCiA_run11_all_local.npy')

VLIC_12 = np.load('VLIC_run12_all_local.npy') 
VLCiD_12 = np.load('VLCiD_run12_all_local.npy')
VLCiA_12 = np.load('VLCiA_run12_all_local.npy')

VLIC_13 = np.load('VLIC_run13_all_local.npy') 
VLCiD_13 = np.load('VLCiD_run13_all_local.npy')
VLCiA_13 = np.load('VLCiA_run13_all_local.npy')

VLIC_14 = np.load('VLIC_run14_all_local.npy') 
VLCiD_14 = np.load('VLCiD_run14_all_local.npy')
VLCiA_14 = np.load('VLCiA_run14_all_local.npy')

VLIC_15 = np.load('VLIC_run15_all_local.npy') 
VLCiD_15 = np.load('VLCiD_run15_all_local.npy')
VLCiA_15 = np.load('VLCiA_run15_all_local.npy')

Time = np.load('Time_run1_all_local.npy')


Connectome_Weights = np.load('CW_run1_all_local.npy',allow_pickle=True)
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
    sns.distplot(spikebins,bins,kde=False,hist=True,color='#492710',hist_kws={'alpha':1},kde_kws={'bw':1.5}) #,kde_kws={'bw':5}
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
    #plt.imshow(stackf,cmap=plt.cm.viridis,vmin=0,vmax=0.5)
    plt.imshow(stackf,cmap=plt.cm.viridis,norm=clr.SymLogNorm(linthresh=0.0001,vmin=0, vmax=0.5))
    plt.colorbar()
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
            if alt == str(VLIC_12):
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
    

Title = 'Tonic Drive Local CiA_CiD Local CiA_MN 15 Runs'
neuron_list = [VLIC_12,VLMN_12,VLCiD_12,VLCiA_12] 
Rasterplot_beatnglide(Time,neuron_list,Title)
spikebins = PlotHist_CalculateSwimFreq(Title)
#Plot_polar(Title)
ChemicalSynapseWeightMatrix(Connectome_Weights, Title)