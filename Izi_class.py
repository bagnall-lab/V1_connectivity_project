#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:46:58 2017

@author: yannwork
"""

#Imports the print function from Python3.
from __future__ import print_function
#Imports the modf function.
from math import modf
#Imports numpy to work with matrix.
import numpy as np
#Imports re to work with string values.
import re
#Imports all ploting utilities.
from pylab import *
import sys

class Izhikevich():
    def __init__(self, a, b, c, d, vmax, dt, x, y):
        self.createNeuron(a, b, c, d, vmax, dt, x, y)
        # a, b, c, d, are the parameters for the membrane potential dynamics
        # vmax is the peak membrane potential of single action potentials
        # x, y are the spatial coordinates of each cell
        
    def createNeuron(self, a, b, c, d, vmax, dt, x, y):
        #Set Neuron constants.
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.vmax = vmax
        self.dt = dt
        self.x = x
        self.y = y
               
    def getNextVal(self, v, u, Stim):
        l = Stim
        if v < self.vmax:
            # ODE eqs
            dv = (0.04*v**2)+5*v+140-u
            vNew = v+(dv+l)*self.dt
            du = self.a*(self.b*v-u)
            uNew = u + self.dt*du
            vOld = v            
        else:
            # Spike
            vOld = self.vmax
            vNew = self.c
            uNew = u + self.d
        
        nVals = np.array([vNew, uNew, vOld])
        return nVals

class LI():
    def __init__(self, R, C, dt, x, y):
        self.createNeuron(R, C, dt, x, y)
        
    def createNeuron(self, R, C, dt, x, y):
        #Set Neuron constants.
        self.R = R
        self.C = C
        self.dt = dt
        self.x = x
        self.y = y
        
        
    def getNextVal(self, v, Stim):
        l = Stim
            # ODE eqs
        dv = (-1/(self.R*self.C))*v+l/self.C
        vNew = v+(dv)*self.dt
        vOld = v
            
        nVals = np.array([vNew, vOld])
        return nVals

class simple_syn():
    def __init__(self, vth, dt, E_rev):
        self.createglusyn(vth, dt, E_rev)
        
    def createglusyn(self, vth, dt, E_rev):
        #Set synapse constants.
        self.vth = vth
        self.dt = dt
        self.E_rev = E_rev 
        
    def getNextVal(self, v1, v2, t):
        if v1 > self.vth:                       # pre-synaptic neuron spikes
            # mPSC
            IsynNew = (self.E_rev-v2)          
        else:
            # no synaptic event
            IsynNew = 0
       
        return IsynNew

    
def Distance(x1, x2, y1, y2):
    d = sqrt(abs(x1-x2)**2 + abs(y1-y2)**2)
    return d
    
class TwoExp_syn():
    def __init__(self, taur, taud, vth, dt, E_rev):
        self.createsyn(taur, taud, vth, dt, E_rev)
        
    def createsyn(self, taur, taud, vth, dt, E_rev):
        #Set synapse constants.
        self.taud = taud
        self.taur = taur
        self.vth = vth
        self.dt = dt
        self.E_rev = E_rev 
        
    def getNextVal(self, v1, v2, IsynA, IsynB):                   
        if v1 > self.vth:                       # pre-synaptic neuron spikes
            # mEPSC
            IsynA = IsynA + (self.E_rev-v2)
            IsynB = IsynB + (self.E_rev-v2)
            dIsynA = (-1/self.taud)*IsynA
            dIsynB = (-1/self.taur)*IsynB
            IsynANew = IsynA +self.dt*(dIsynA)
            IsynBNew = IsynB +self.dt*(dIsynB)
            IsynNew = (IsynANew - IsynBNew)        
        else:
            # no synaptic event
            dIsynA = (-1/self.taud)*IsynA
            dIsynB = (-1/self.taur)*IsynB
            IsynANew = IsynA +self.dt*(dIsynA)
            IsynBNew = IsynB +self.dt*(dIsynB)
            IsynNew = (IsynANew - IsynBNew)
           
        return IsynNew, IsynANew, IsynBNew
    

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    result = result[int(result.size/2):]
    return result/np.max(result)

def Xcorr(x, y):
    result = np.correlate(x, y, mode='full')
    result = result[int(result.size/2):]
    return result/np.max(result)