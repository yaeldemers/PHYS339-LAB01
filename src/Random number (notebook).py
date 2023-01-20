#!/usr/bin/env python
# coding: utf-8

# In[59]:


#------------Import statements-----------#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import random as r
import seaborn as sns
sns.set()


# In[60]:


# Problem 7.1


# In[61]:


#------------Global variables-----------#
A=2 # Obtained by integrating PDF over 0-1
nsamp = 1000
values=np.zeros(nsamp)
trials=list(range(nsamp))
count_results= np.zeros((10,20))


# In[62]:


#----------- Linear PDF & CDF ----------#
def linearModelPDF(A, x):
    x = np.array(x)
    return A*x

# CDF is obtained by integrating PDF from -inf to inf
def linearModelCDF(A, x): 
    x = np.array(x)
    return A*(x**2)/2


# In[63]:


#--------- Apply reverse rule ---------#
def reverse(data):
    for i in trials:
        data[i] = np.sqrt(2*r.random()/A)


# In[64]:


#----- Core of exp to be repeated -----#
def experiment(exp_name, save, stats=None):    
    reverse(values)
    
    binNumber = 20
    cumBinNumber = 100

    # Obtain histogram with number of bins equal to binNumber--this is to compare to the PDF
    histValues1, binEdges1 = np.histogram(values, binNumber)
    
    # Obtain histogram with number of bins equal to cumbinNumber--this is to compare to the CDF
    histValues2, binEdges2 = np.histogram(values, cumBinNumber)
    cumHistValues = np.cumsum(histValues2)/nsamp

    # Need to initialize new arrays binCenterHist and binCenterCumHist
    binCenterHist = np.zeros(len(binEdges1)-1)
    binCenterCumHist = np.zeros(len(binEdges2)-1)

    # Convert bin edges to bin center position
    for i in range(len(binEdges1) - 1):
        binCenterHist[i] = 0.5 * (binEdges1[i] + binEdges1[i + 1])
    for i in range(len(binEdges2) - 1):
        binCenterCumHist[i] = 0.5 * (binEdges2[i] + binEdges2[i + 1])

    # Find bin widths
    binWidthHist=np.diff(binEdges1)

    # Random values plot
    norm = nsamp*binWidthHist[0]
    f1 = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(trials, values, 'o', markersize=0.75)
    plt.xlabel('trials')
    plt.ylabel('values')

    # PDF plot
    plt.subplot(2, 2, 2)
    plt.plot(binCenterHist, histValues1, 'o', markersize=3)
    plt.plot(binCenterHist, norm*linearModelPDF(A, binCenterHist), 'k', linewidth=0.75)
    if stats is not None:
        plt.errorbar(binCenterHist, histValues1, yerr=stats, fmt="o", markersize=0, linewidth=1)
    plt.xlabel('values')
    plt.ylabel('counts')

    # CDF plot
    plt.subplot(2, 2, 3)
    plt.plot(binCenterCumHist, cumHistValues, 'o', markersize=3)
    plt.plot(binCenterCumHist, linearModelCDF(A, binCenterCumHist), 'k', linewidth=0.75)
    plt.xlabel('value')
    plt.ylabel('cumulative counts')

    plt.tight_layout()
    
    if save:
        plt.savefig(exp_name+'Plot.png')
        plt.show()

    # These commands output the histogram and cumulative histogram results to csv files
    # for plotting
    np.savetxt(exp_name+'Hist.csv',
    np.transpose([binCenterHist, histValues1]), delimiter = ",")
    np.savetxt(exp_name+'CumHist.csv',
    np.transpose([binCenterCumHist, cumHistValues]), delimiter = ",")
    
    return histValues1


# In[65]:


#--------- Repeating the experiment ---------#
for i in range(10):
    r.seed(i) # for experiment reproducibility
    exp_name = 'trial'+str(i)
    count_results[i] = experiment(exp_name, save=False)


# In[67]:


means, stds = np.mean(count_results, axis=0), np.std(count_results, axis=0)        

r.seed(11)

out = experiment('mainExperiment', save=True, stats=stds)


# In[68]:


# 7.2


# In[69]:


nsamp = 1000
A=1
values=np.zeros(nsamp)
trials=list(range(nsamp))
sigma = 1


# In[70]:


moose = np.zeros(nsamp)
rs = np.zeros(nsamp)
thetas = np.zeros(nsamp)

# Setting mu's and thethas
for i in range(len(values)):
    moose[i] = -(1/A) * np.log(1 - r.random())
    thetas[i] = r.random()*2*np.pi

rs = np.sqrt(2*moose*sigma**2)

ys = rs*np.sin(thetas)

xs = rs*np.cos(thetas)

plt.plot(xs, ys, 'o', markersize=3)
plt.title('1000 data points - gaussian distributed on x and y')
plt.show()


# In[71]:


# 1000 data points generated from gaussian distribution on the dependent variable
uniform_xs = np.arange(0, 1000, 1)
plt.plot(uniform_xs, ys, 'o', markersize=3)
plt.title(' 1000 data points - gaussian distributed')
plt.show()


# In[118]:


# Plot the histogram
plt.vlines(x = 0, ymin = 0, ymax = 0.45,
           color = 'orange',
           label = '$\mu$',
           linestyle='dashed')

plt.vlines(x = 1, ymin = 0, ymax = 0.45,
           color = 'green',
           label = '$\sigma$',
           linestyle='dashed')

plt.vlines(x = -1, ymin = 0, ymax = 0.45,
           color = 'green',
           label = '$-\sigma$',
           linestyle='dashdot')

plt.hist(ys, bins = 25, density=True)

# Plot the normal curve with mu=0 and std=1
mu, std = 0, 1
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = stats.norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2, label='Norm(0,1)')
plt.title('Generated data following a gaussian distribution with a normal curve with $\mu=0$ and $\sigma=1$')
plt.ylabel('Density')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




