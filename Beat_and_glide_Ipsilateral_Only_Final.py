    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
"""

# Types of neurons in models: 
        
        #Single Coiling Model (ICs, MN, and CoLA):
        # Ipsilateral Caudal interneurons (IC neurons) aka Pacemaker neurons 
        # Motor neurons 
        # Commissural longitudninal ascending neurons (COLAs) aka Commissural interneurons **** Commisurral Bifurcating Longitudinal Neurons (CoBL) aka V0d aka dl6

        #Multiple coiling model (ICs, MN, CoLA, and CoE): 
        # Ipsilateral Caudal interneurons (IC neurons) aka Pacemaker neurons 
        # Motor neurons 
        # Commissural longitudninal ascending neurons (COLAs) aka Commissural interneurons 
        # CoEs are Commissural excitatory neurons (CENs) or V0vs

        #Burstswimming model (ICs, MN, CoLA, CoE, and CiD):
        # Ipsilateral Caudal interneurons (IC neurons) aka Pacemaker neurons 
        # Motor neurons 
        # Commissural longitudninal ascending neurons (COLAs) aka Commissural interneurons aka CINs
        # CoEs are Commissural excitatory neurons (CENs) or V0vs
        # Circumferential descending neurons (CiD) aka V2A neurons aka IEDs
        
        # Beat and glide model: 
        # Ipsilateral Caudal interneurons (IC neurons) aka Pacemaker neurons 
        # Motor neurons 
        # Commissural longitudninal ascending neurons (COLAs) aka Commissural interneurons 
        # CoEs are Commissural excitatory neurons (CENs) or V0vs
        # Circumferential descending neurons (CiD) aka V2A neurons aka IEDs
        # CiA or V1 neurons aka IIAs

        #V2B and V1 Mohini model: 
        # Ipsilateral Caudal interneurons (IC neurons) aka Pacemaker neurons 
        # Motor neurons 
        # Commissural longitudninal ascending neurons (COLAs) aka Commissural interneurons 
        # CoEs are Commissural excitatory neurons (CENs) or V0vs
        # Circumferential descending neurons (CiD) aka V2A neurons 
        # CiA or V1 neurons aka IIAs 
        # V2B neurons or ipsilateral inhibitory descending neurons (IIDs)

from random import *
from Izi_class import * # Where definition of single cell models are found based on Izhikevich models

try:
    from progressbar import ProgressBar
    pbar = ProgressBar()
except ImportError:
    print("Please run: pip install progressbar")


# This function sets up the connectome and runs a simulation for the time tmax.
# rand is a seed for a random function
# stim0 is a constant for scaling the drive to IC neurons
# sigma is a variance for gaussian randomization of the gap junction coupling
# dt is the time step size
# E_glu and E_gly are the reversal potential of glutamate and glycine respectively
# c is the transmission speed
# nIC, nMN, nCoLA, and nMuscle is the number of IC, MN, CoLA, CoE, CiD, CiA and Muscle cells
# R_str is an indication of whether glycinergic synapses are present or blocked by strychnine (str). Ranges from 0 to 1. 
#    1: they are present; 0: they are all blocked

def connectome_beat_glide_ipsilateral(rand=0, stim0=0.35, sigma=0,
                          tmax=1000, dt=0.1, E_glu=0, E_gly=-70, c=0.55,
                          nIC=5, nMN=15, nCiD=12, nCiA=10, 
                          R_str=1.0):
    seed(rand)
    ## Declare constants

    tmax += 200 # add 200 ms to give time for IC to settle.  This initial 200 ms is not used in the data analysis
    
    nmax = int(tmax/dt) 
    
    ## This model is based on Izhikevich neuron type:
    # Suggested parameter values for different firing types are provided below
    # tonic bursting a:0.02 b:0.25 c:-50 d:2.0 vmax=-10 (IC you can increase d until 4.0)
    # tonic spiking a:0.02 b:0.2 c:-65 d:6.0 (pMN, CiD)
    # phasic spiking a:0.02 b:0.25 c:-65 d:6.0 (single spike response)
    # phasic bursting a:0.02 b:0.25 c:-55 d:0.05
    #spike frequency adaptation a:0.01 b:0.1 c:-65.0 d:8.0
    #acomodation a:0.0 b:0.19 c:-60.0 d:2.8
    #bistability a:0.1 b:0.26 c:-57.8 d:0.0
    
    ## Declare Neuron Types
    
    L_IC = [ Izhikevich(0.02,0.25,-50,2.0,-10,dt,1.0,-1) for i in range(nIC)]
    # L_MN = [ Izhikevich(0.0175,0.1,-53,8.0, 0,dt, 5.0+1.6*i,-1) for i in range(nMN)]      
    L_MN = [ Izhikevich(0.1,0.25,-53,6.0, 0,dt, 5.0+1.6*i,-1) for i in range(nMN)]      
    L_CiD = [ Izhikevich(0.1,0.25,-53,6.0, 0,dt, 5.1+1.6*i,-1) for i in range(nCiD)] 
    L_CiA = [ Izhikevich(0.2,0.25,-53,6.0, 0,dt, 8.1+1.6*i,-1) for i in range(nCiA)]
    
    ## Declare Synapses
    
    L_glysyn_CiA_CiD = [ TwoExp_syn(0.5, 3.0, -15, dt, E_gly) for i in range (nCiA*nCiD)] 
    L_glysyn_CiA_IC = [ TwoExp_syn(0.5, 3.0, -15, dt, E_gly) for i in range (nCiA*nIC)]
    L_glysyn_CiA_MN = [ TwoExp_syn(0.5, 3.0, -15, dt, E_gly) for i in range (nCiA*nMN)]
    
    L_glusyn_CiD_CiD = [TwoExp_syn(0.25, 1.0, -15, dt, E_glu) for i in range (nCiD*nCiD)]
    L_glusyn_CiD_MN = [TwoExp_syn(0.25, 1.0, -15, dt, E_glu) for i in range (nCiD*nMN)]
    L_glusyn_CiD_CiA = [TwoExp_syn(0.25, 1.0, -15, dt, E_glu) for i in range (nCiD*nCiA)]

    ## Declare Storage tables
    
    Time =zeros(nmax)
    
    VLIC =zeros((nIC, nmax))
    VLMN =zeros((nMN, nmax))
    VLCiD = zeros ((nCiD,nmax))
    VLCiA = zeros ((nCiA,nmax))
    
    #storing synaptic currents
    
    #gly
    LSyn_CiA_CiD = zeros((nCiA*nCiD,3))
    LSyn_CiA_IC = zeros((nCiA*nIC,3))
    LSyn_CiA_MN = zeros((nCiA*nMN,3))
    
    #glu

    LSyn_CiD_CiD = zeros((nCiD*nCiD,3))
    LSyn_CiD_MN = zeros((nCiD*nMN,3))
    LSyn_CiD_CiA = zeros((nCiD*nCiA,3))
    
    #Gap
    LSGap_IC_IC = zeros((nIC,nIC))
    LSGap_IC_MN = zeros((nIC,nMN))      
    LSGap_IC_CiD = zeros((nIC,nCiD))
    LSGap_IC_CiA = zeros((nIC,nCiA))
    LSGap_MN_IC = zeros((nMN,nIC))
    LSGap_MN_MN = zeros((nMN,nMN))
    LSGap_MN_CiD = zeros((nMN,nCiD))

    LSGap_CiD_IC = zeros((nCiD,nIC))
    LSGap_CiD_CiD = zeros((nCiD,nCiD))
    LSGap_CiD_MN = zeros((nCiD,nMN))
    LSGap_CiA_CiA = zeros((nCiA,nCiA))
    LSGap_CiA_IC = zeros((nCiA,nIC))
    
    
        #Synaptic weight
        
    #gly


    LW_CiA_CiD = zeros((nCiA,nCiD))      

    LW_CiA_IC = zeros((nCiA,nIC))      


    LW_CiA_MN = zeros((nCiA,nMN))


    
    #glu


    LW_CiD_CiD = zeros((nCiD,nCiD))

    LW_CiD_MN = zeros((nCiD,nMN))


    LW_CiD_CiA = zeros((nCiD,nCiA))
    
    
        #Gap junctions coupling
    LGap_IC_MN = zeros((nIC,nMN))      

    LGap_IC_IC = zeros((nIC,nIC))

    LGap_IC_CiD = zeros((nIC,nCiD))
    LGap_IC_CiA = zeros((nIC,nCiA))
    LGap_MN_MN = zeros((nMN,nMN))


    LGap_CiD_CiD = zeros((nCiD,nCiD))
    LGap_CiD_MN = zeros((nCiD,nMN))
    LGap_CiA_CiA = zeros((nCiA,nCiA))
    
        #res
    resLIC=zeros((nIC,3))

    resLMN=zeros((nMN,3))



    resLCiD = zeros((nCiD,3))
    resRCiD = zeros((nCiD,3))
    resLCiA = zeros((nCiA,3))
    resRCiA = zeros((nCiA,3))

    
    stim= zeros (nmax)
    stim2= zeros (nmax)
    
    #remove in the final version
    stim_test1 = zeros(nmax)
    stim_test2 = zeros(nmax)
    stim_test3 = zeros(nmax)
    stim_test4 = zeros(nmax)
    stim_test5 = zeros(nmax)
    
        #Distance Matrix 
    LD_IC_MN = zeros((nIC, nMN))        #for Gap

    LD_IC_CiD = zeros((nIC,nCiD))
    LD_IC_CiA = zeros((nIC,nCiA))     
    LD_MN_MN = zeros((nMN,nMN))


    
    LD_CiD_CiD = zeros((nCiD,nCiD))     
    LD_CiD_MN = zeros((nCiD,nMN))       
    
 
    LD_CiD_CiA = zeros((nCiD,nCiA))   
 
    
    LD_CiA_IC = zeros((nCiA,nIC))   

    LD_CiA_MN = zeros((nCiA, nMN))
    LD_CiA_CiA = zeros((nCiA,nCiA))
    
    
    ## Compute distance between Neurons
    
    #LEFT
    for k in range (0, nIC):
        for l in range (0, nMN):
            LD_IC_MN[k,l] = Distance(L_IC[k].x,L_MN[l].x,L_IC[k].y,L_MN[l].y)
            
    for k in range (0, nIC):
        for l in range (0, nCiD):
            LD_IC_CiD[k,l] = Distance(L_IC[k].x,L_CiD[l].x,L_IC[k].y,L_CiD[l].y)

    for k in range (0, nIC):
        for l in range (0, nCiA):
            LD_IC_CiA[k,l] = Distance(L_IC[k].x,L_CiA[l].x,L_IC[k].y,L_CiA[l].y)
            
    for k in range (0, nMN):
        for l in range (0, nMN):
            LD_MN_MN[k,l] = Distance(L_MN[k].x,L_MN[l].x,L_MN[k].y,L_MN[l].y)
 
    for k in range (0, nCiD):
        for l in range (0, nCiD):
            LD_CiD_CiD[k,l] = Distance(L_CiD[k].x,L_CiD[l].x,L_CiD[k].y,L_CiD[l].y)
            
    for k in range (0, nCiD):
        for l in range (0, nMN):
            LD_CiD_MN[k,l] = Distance(L_CiD[k].x,L_MN[l].x,L_CiD[k].y,L_MN[l].y)
            
    for k in range (0, nCiD):
        for l in range (0, nCiA):
            LD_CiD_CiA[k,l] = Distance(L_CiD[k].x,L_CiA[l].x,L_CiD[k].y,L_CiA[l].y)

    for k in range (0, nCiA):
        for l in range (0, nIC):
            LD_CiA_IC[k,l] = Distance(L_CiA[k].x,L_IC[l].x,L_CiA[k].y,L_IC[l].y)

    for k in range (0, nCiA):
        for l in range (0, nMN):
            LD_CiA_MN[k,l] = Distance(L_CiA[k].x,L_MN[l].x,L_CiA[k].y,L_MN[l].y) 

    for k in range (0, nCiA):
        for l in range (0, nCiA):
            LD_CiA_CiA[k,l] = Distance(L_CiA[k].x,L_CiA[l].x,L_CiA[k].y,L_CiA[l].y)
            
    
    ## Compute synaptic weights and Gaps
    
    #LEFT
    IC_IC_gap_weight = 0.0001  # default 0.001
    for k in range (0, nIC):
        for l in range (0, nIC):
            if (k!= l):                                         #because kernel
                LGap_IC_IC[k,l] = IC_IC_gap_weight 
            else:
                LGap_IC_IC[k,l] = 0.0

    IC_CiD_gap_weight = 0.008 #default 0.0002
    for k in range (0, nIC):
        for l in range (0, nCiD):
            if (0.2<LD_IC_CiD[k,l]<10 and L_IC[k].x < L_CiD[l].x):   #because descending
                LGap_IC_CiD[k,l] = IC_CiD_gap_weight 
            else:
                LGap_IC_CiD[k,l] = 0.0

    #chem syn
    
    CiD_CiD_syn_weight = 0.1 #default 0.02
    for k in range (0, nCiD):
        for l in range (0, nCiD):
            if (0.2<LD_CiD_CiD[k,l]<10 and L_CiD[k].x < L_CiD[l].x): #descending
                LW_CiD_CiD[k,l] = CiD_CiD_syn_weight
            else:
                LW_CiD_CiD[k,l] = 0.0
    
    CiD_MN_syn_weight =  0.1 # default 0.75
    for k in range (0, nCiD):
        for l in range (0, nMN):
            if ((0.2<LD_CiD_MN[k,l]<10 and L_CiD[k].x < L_MN[l].x) or (0.2<LD_CiD_MN[k,l]<4 and L_CiD[k].x > L_MN[l].x)): #descending and short ascending branch
                LW_CiD_MN[k,l] = CiD_MN_syn_weight
            else:
                LW_CiD_MN[k,l] = 0.0

    CiD_CiA_syn_weight = 0.2 #default = 0.12              
    for k in range (0, nCiD):
        for l in range (0, nCiA):
            if (0.2<LD_CiD_CiA[k,l]<10 and L_CiD[k].x < L_CiA[l].x):   #descending
                LW_CiD_CiA[k,l] = CiD_CiA_syn_weight 
            else:
                LW_CiD_CiA[k,l] = 0.0           

    #CiA_CiD_syn_weight = 0.0
       
    for k in range (0, nCiA):
        for l in range (0, nCiD):
            if (0.2<LD_CiD_CiA[l,k]<5.0 and L_CiD[l].x < L_CiA[k].x):    #descending 0.2<LD_CiD_CiA[l,k]<10.0
                CiA_CiD_syn_weight = abs(0.5 * gauss(1,0.25))
                LW_CiA_CiD[k,l] = CiA_CiD_syn_weight
            else:
                LW_CiA_CiD[k,l] = 0.0

    indx = np.argwhere(LW_CiA_CiD>0)
    sum_CIA_CiD = sum(LW_CiA_CiD)
    print(sum_CIA_CiD)

    defaultsum = 10.5 

    for x,y in indx:
        LW_CiA_CiD[x,y]=LW_CiA_CiD[x,y]/(sum_CIA_CiD/defaultsum)

    print(sum(LW_CiA_CiD))

    #CiA_MN_syn_weight = 0.0

    for k in range (0, nCiA):
        for l in range (0, nMN):
            if (0.2<LD_CiA_MN[k,l]<5.0 and L_MN[l].x < L_CiA[k].x):    #descending 0.2<LD_CiD_CiA[l,k]<10.0
                CiA_MN_syn_weight = abs(0.5 * gauss(1,0.25)) #Best so far 1.25
                LW_CiA_MN[k,l] = CiA_MN_syn_weight
            else:
                LW_CiA_MN[k,l] = 0.0

    indx = np.argwhere(LW_CiA_MN>0)
    sum_CIA_MN = sum(LW_CiA_MN)
    print(sum_CIA_MN)

    # #defaultsum = 10.5 

    for x,y in indx:
        LW_CiA_MN[x,y]=LW_CiA_MN[x,y]/(sum_CIA_MN/defaultsum)

    print(sum(LW_CiA_MN))
   
    CiA_IC_syn_weight =  0.5 # default 0.005 
    for k in range (0, nCiA):
        for l in range (0, nIC):
            if (0.2<LD_CiA_IC[k,l]<40 and L_IC[l].x < L_CiA[k].x):    #ascending
                LW_CiA_IC[k,l] = CiA_IC_syn_weight 
            else:
                LW_CiA_IC[k,l] = 0.0
    

    Connectome_Weights = {'LGap_IC_IC':LGap_IC_IC, 'LGap_IC_MN': LGap_IC_MN, 
                       'LGap_IC_CiD':LGap_IC_CiD,  'LGap_MN_MN':LGap_MN_MN,  
                       'LGap_CiD_CiD': LGap_CiD_CiD, 'LGap_CiD_MN': LGap_CiD_MN, 
                       'LW_CiD_CiD':LW_CiD_CiD, 'LW_CiD_MN':LW_CiD_MN,
                       'LW_CiD_CiA':LW_CiD_CiA, 'LW_CiA_CiD':LW_CiA_CiD,
                      'LW_CiA_IC': LW_CiA_IC,  'LW_CiA_MN':LW_CiA_MN, 'LGap_CiA_CiA':LGap_CiA_CiA,'LGap_IC_CiA':LGap_IC_CiA}
    
    ## init membrane potential values           
    for k in range (0, nIC):
        resLIC[k,:] = L_IC[k].getNextVal(-65,-16,-65)
        VLIC[k,0] = resLIC[k,0]
    
    for k in range (0, nMN):
        resLMN[k,:] = L_MN[k].getNextVal(-70,-14,-70)
        VLMN[k,0] = resLMN[k,0]
        
    for k in range (0, nCiD):
        resLCiD[k,:] = L_CiD[k].getNextVal(-64,-16,-64)
        VLCiD[k,0] = resLCiD[k,0]
        
    for k in range (0, nCiA):
        resLCiA[k,:] = L_CiA[k].getNextVal(-64,-16,-64)
        VLCiA[k,0] = resLCiA[k,0]
        
    ## ODE Solving (Time loop)
    
    for t in pbar(range (0, nmax)):
        Time[t]=dt*t
        
        # if not(Time[t] % 500) and (Time[t]>20):
            
        #     fig, ax = plt.subplots(2,1, sharex=True, figsize=(15, 15)) 
        #     cmapL = matplotlib.cm.get_cmap('Blues')
        #     cmapR = matplotlib.cm.get_cmap('Reds')
        #     ax[0].plot([0], [0], c=cmapL(0.5))

        #     for k in range (0, nMN):
        #         ax[0].plot(Time, VLMN[k,:], c=cmapL((k+1)/nMN)) # adding a color gradiant, darker color -> rostrally located
        #         ax[0].plot(Time, VRMN[k,:], c=cmapR((k+1)/nMN))
        #     plt.xlabel('Time (ms)')
        #     plt.xlim([0, Time[t]])
        #     plt.show()
        
        if t > 2000: #YANN, we'll want to clarify this
            stim2[t] = stim0
        
        ## Synaptic inputs
                   
    
        for k in range (0, nCiD):
            for l in range (0, nCiD):
                LSyn_CiD_CiD[nCiD*k+l,:] = L_glusyn_CiD_CiD[nCiD*k+l].getNextVal(VLCiD[k,t-int(LD_CiD_CiD[k,l]/(dt*c))], VLCiD[l,t-1], LSyn_CiD_CiD[nCiD*k+l,1], LSyn_CiD_CiD[nCiD*k+l,2])  #Contralateral
    
        for k in range (0, nCiD):
            for l in range (0, nMN):
                LSyn_CiD_MN[nMN*k+l,:] = L_glusyn_CiD_MN[nMN*k+l].getNextVal(VLCiD[k,t-int(LD_CiD_MN[k,l]/(dt*c))], VLMN[l,t-1], LSyn_CiD_MN[nMN*k+l,1], LSyn_CiD_MN[nMN*k+l,2])  #Contralateral
    
        for k in range (0, nCiD):
            for l in range (0, nCiA):
                LSyn_CiD_CiA[nCiA*k+l,:] = L_glusyn_CiD_CiA[nCiA*k+l].getNextVal(VLCiD[k,t-int(LD_CiD_CiA[k,l]/(dt*c))], VLCiA[l,t-1], LSyn_CiD_CiA[nCiA*k+l,1], LSyn_CiD_CiA[nCiA*k+l,2])
                
        for k in range (0, nCiA):
            for l in range (0, nCiD):
                LSyn_CiA_CiD[nCiD*k+l,:] = L_glysyn_CiA_CiD[nCiD*k+l].getNextVal(VLCiA[k,t-int(LD_CiD_CiA[l,k]/(dt*c))], VLCiD[l,t-1], LSyn_CiA_CiD[nCiD*k+l,1], LSyn_CiA_CiD[nCiD*k+l,2])
            
        for k in range (0, nCiA):
            for l in range (0, nIC):
                LSyn_CiA_IC[nIC*k+l,:] = L_glysyn_CiA_IC[nIC*k+l].getNextVal(VLCiA[k,t-int(LD_CiA_IC[k,l]/(dt*c))], VLIC[l,t-1], LSyn_CiA_IC[nIC*k+l,1], LSyn_CiA_IC[nIC*k+l,2])
                
        for k in range (0, nCiA):
            for l in range (0, nMN):
                LSyn_CiA_MN[nMN*k+l,:] = L_glysyn_CiA_MN[nMN*k+l].getNextVal(VLCiA[k,t-int(LD_CiA_MN[k,l]/(dt*c))], VLMN[l,t-1], LSyn_CiA_MN[nMN*k+l,1], LSyn_CiA_MN[nMN*k+l,2])
                
        for k in range (0, nIC):
            for l in range (0, nIC):   
                LSGap_IC_IC[k,l] = LGap_IC_IC[k,l]*(VLIC[k,t-1]-VLIC[l,t-1])
                                                
        for k in range (0, nIC):
            for l in range (0, nCiD):   
                LSGap_IC_CiD[k,l] = LGap_IC_CiD[k,l]*(VLIC[k,t-int(LD_IC_CiD[k,l]/(dt*c))]-VLCiD[l,t-1])
                        
        for k in range (0, nCiD):
            for l in range (0, nIC):   
                LSGap_CiD_IC[k,l] = LGap_IC_CiD[l,k]*(VLCiD[k,t-int(LD_IC_CiD[l,k]/(dt*c))]-VLIC[l,t-1])
                  
        ## Membrane potentials
        
        for k in range (0, nIC):
            if t < 500: #YANN, we'll want to clarify this
                IsynL= 0.0
            else:
                IsynL = sum(LSyn_CiA_IC[nIC*l+k,0]*LW_CiA_IC[l,k]*R_str for l in range (0, nCiA))
                
                
            resLIC[k,:] = L_IC[k].getNextVal(resLIC[k,0],resLIC[k,1],stim2[t] - sum(LSGap_IC_IC[k,:]) + sum(LSGap_IC_IC[:,k]) + IsynL) #+ sum(LSGap_MN_IC[:,k]) - sum(LSGap_IC_MN[k,:]) + sum(LSGap_CiD_IC[:,k]) - sum(LSGap_IC_CiD[k,:]) #
            VLIC[k,t] = resLIC[k,0]
            
        for k in range (0, nMN):
            if t < 500: #YANN, we'll want to clarify this
                IsynL= 0.0
            else:
                IsynL = sum(LSyn_CiD_MN[nMN*m+k,0]*LW_CiD_MN[m,k] for m in range (0, nCiD)) + sum(LSyn_CiA_MN[nMN*l+k,0]*LW_CiA_MN[l,k]*R_str for l in range (0, nCiA))
                            
            resLMN[k,:] = L_MN[k].getNextVal(resLMN[k,0],resLMN[k,1], IsynL) #- sum(LSGap_MN_IC[k,:]) + sum(LSGap_IC_MN[:,k]) - sum(LSGap_MN_MN[k,:]) + sum(LSGap_MN_MN[:,k]) - sum(LSGap_MN_CiD[k,:]) + sum(LSGap_CiD_MN[:,k]) 
            VLMN[k,t] = resLMN[k,0]
            
        for k in range (0, nCiD):
            if t < 2000: #YANN, we'll want to clarify this, why are CiD's not getting a drive before 2000, or is it 200 ms?
                IsynL= 0.0
            else:
                IsynL = sum(LSyn_CiD_CiD[nCiD*m+k,0]*LW_CiD_CiD[m,k] for m in range (0, nCiD)) + sum(LSyn_CiA_CiD[nCiD*p+k,0]*LW_CiA_CiD[p,k]*R_str for p in range (0, nCiA))
            if k<20:
                IsynL = IsynL + stim2[t-180-32*k] * (nCiD-k)/nCiD        # Tonic drive for the all CiDs # (nCiD-k)/nCiD is to produce a decreasing gradient of descending drive # 32*k represents the conduction delay, which is 1.6 ms according to McDermid and Drapeau JNeurophysiol (2006). Since we consider each somite to be two real somites, then 16*2 
                
                #YANN, we'll want to clarify the gaussians here
            resLCiD[k,:] = L_CiD[k].getNextVal(resLCiD[k,0],resLCiD[k,1], - sum(LSGap_CiD_IC[k,:]) - sum(LSGap_IC_CiD[:,k]) + IsynL) #- sum(LSGap_CiD_CiD[k,:]) + sum(LSGap_CiD_CiD[:,k]) - sum(LSGap_CiD_MN[k,:])+ sum(LSGap_MN_CiD[:,k])          
            VLCiD[k,t] = resLCiD[k,0]
    
    
        for k in range (0, nCiA):
            if t < 500:
                IsynL= 0.0
            else:
                IsynL = sum(LSyn_CiD_CiA[nCiA*m+k,0]*LW_CiD_CiA[m,k] for m in range (0, nCiD))

            resLCiA[k,:] = L_CiA[k].getNextVal(resLCiA[k,0],resLCiA[k,1], IsynL) #- sum(LSGap_CiA_CiA[k,:]) + sum(LSGap_CiA_CiA[:,k]) - sum(LSGap_CiA_IC[k,:]) - sum(LSGap_IC_CiA[:,k])   
            VLCiA[k,t] = resLCiA[k,0]
            
            

    VLICnew = VLIC[:,2000:]
    
    VLMNnew = VLMN[:,2000:]
    
    VLCiDnew = VLCiD[:,2000:]
    
    VLCiAnew = VLCiA[:,2000:]
    
    Timenew = Time[2000:]-Time[2000:][0]
    
    
    return (VLICnew), (VLMNnew), (VLCiDnew), (VLCiAnew), Timenew, Connectome_Weights, sum(LW_CiA_MN), sum(LW_CiA_CiD)

