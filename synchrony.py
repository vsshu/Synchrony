# -*- coding: utf-8 -*-
"""
Synchrony and mimicry in dyadic human interactions

Vivian Shan-shan Hu
Core Topics AI 
Tilburg University
10/12/21

"""
#%%
import os
from xml.etree import ElementTree

import numpy as np
import pandas as pd

import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from pyunicorn.timeseries import CrossRecurrencePlot

#Create aggregation file
#aggdf = pd.DataFrame()
#%%
# =============================================================================
# Import and read files
# =============================================================================
path = r'C:\Users\gebruiker\Documents\Synch Mim\ShakeFive2\Meta'
files = os.listdir(path)
#print(len(files))
filenr = 54
file1 = os.path.join(path, files[filenr])
tree = ElementTree.parse(file1)
root = tree.getroot()

#%%
# =============================================================================
# Extract data
# =============================================================================
skeleton0 = []
skeleton1 = []

for skelet in root.findall('.//skeletons'):
    skeleton0.append(skelet[0][5][11][2].text)
    try:
        skeleton1.append(skelet[1][5][11][2].text)
    except Exception: 
        skeleton1.append('0 0')

# =============================================================================
# Modify Data
# =============================================================================
def modify(rawdata):
    cleaned = []
    quo = []
    for line in rawdata:
        cleaned.append(line.split())
    
    cleanx = np.float_([item[0] for item in cleaned])
    cleany = np.float_([item[1] for item in cleaned])  
    
    for i in range(len(cleanx)-10):
        dist = math.sqrt((cleanx[i+10] - cleanx[i])**2 + (cleany[i+10] - cleany[i])**2)
        quo.append(dist)
        
    return quo

df = pd.DataFrame({'zero':modify(skeleton0), 'one':modify(skeleton1)})
shuffle = df.sample(frac=1)

#aggdf.to_csv('bump_aggdata.csv')
#aggdf.to_csv('hug_aggdata.csv')

# =============================================================================
# Plot Time Series
# =============================================================================
def ts():
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(df['zero'], color='b', label='zero')
    ax[0].legend()
    ax[1].plot(df['one'], color='r', label='one')
    ax[1].legend()
    return

def tfautocor():
    for variable in df.columns:
        pd.plotting.autocorrelation_plot(df[variable], label = variable)
    return

print(ts())
#print(tfautocor())

# =============================================================================
# Scatter Plot
# =============================================================================
def scatter(plot = 0):
    plt.title("Scatter plot");
    if plot == 0:
        pd.plotting.lag_plot(df['zero'], label='zero')
    if plot == 1:
        pd.plotting.lag_plot(df['one'], label='one')
    plt.legend()
 
#print(scatter(0))

# =============================================================================
# Measurement       TIME LAGGED CROSS CORRELATION
# =============================================================================
def crosscorr(datax, datay, lag=0, wrap=False):
    return datax.corr(datay.shift(lag))

def TLCC():
    d0 = df['zero']
    d1 = df['one']
    #ds = shuffle['one']
    
    rs = [crosscorr(d0,d1, lag) for lag in range(-int(100),int(100))]   
    offset = np.floor(len(rs)/2)-np.argmax(rs)
    f,ax=plt.subplots()
    ax.plot(rs)
    
    ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
    ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
    
    ax.set(title=f'Time lagged cross correlation {filenr} \n Lag Offset = {offset} frames', xlabel='Offset',ylabel='Pearson r')
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels([-100, -50, 0, 50, 100]);
    plt.legend()
    
    lags = np.arange(-(100), (100), 1) 
    print("lag:", lags[np.argmax(rs)], "correlation:", np.max(rs))
    
    #create aggregation file 
    #aggdf[filenr] = rs
    return

print(TLCC())

#%%
# =============================================================================
# Aggregated Bump+Shake Data
# =============================================================================
def aggr():
    bump = pd.read_csv('bump_aggdata.csv', usecols=range(1,9))
    hug = pd.read_csv('hug_aggdata.csv', usecols=range(1,9))
    y=range(0,200)
    
    f,ax=plt.subplots(figsize=(10,5))
    ax.set(title='Aggregated Time Lagged Cross Correlation')
    ax.errorbar(y, bump.mean(axis=1), bump.std(axis=1), label='Fist bump')
    ax.errorbar(y, hug.mean(axis=1), hug.std(axis=1), label='hug')
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels([-100, -50, 0, 50, 100]);
    plt.legend()
    return

#print(aggr())

#%%
# =============================================================================
# Measurement      CROSS RECURRENCE ANALYSIS
# =============================================================================
def CRA():
    # Array-ify
    data2D = shuffle.to_numpy()
    #data2Ds = df.to_numpy()

    
    EPS = 0.001  # Fixed threshold
    METRIC = "supremum"
        
    cp = CrossRecurrencePlot(data2D[:,0], data2D[:,1], metric=METRIC, normalize=True, threshold=EPS)
    #rp = RecurrencePlot(data2D[:,0], dim=DIM, tau=TAU, metric=METRIC, normalize=False, threshold=EPS)

    ##Calculate some standard RQA measures
    RR = cp.cross_recurrence_rate()
    DET = cp.determinism(l_min=2)
    LAM = cp.laminarity(v_min=2)
    Lmax = cp.max_diaglength()
    print("Recurrence rate:", RR, "Determinism:", DET, "Laminarity:", LAM,
          "Maximum line:", Lmax, )#'shuffle')
    
    
    # ax.matshow(cr.recurrence_matrix())
    f, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cp.recurrence_matrix(), cmap=cm.Blues)
    ax.set(title=f"Cross recurrence plot {filenr}", xlabel="$zero$",ylabel="$one$")
    plt.show
    return

print(CRA())