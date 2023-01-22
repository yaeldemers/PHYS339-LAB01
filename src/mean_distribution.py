# PART 7.3 - STUDYING MEAN DISTRIBUTIONS

#------------Import statements-----------#
from uniform_model import UNI
from gaussian_model import GAU
from linear_model import LIN
import matplotlib.pyplot as plt
import numpy as np
import random as r
import seaborn as sns
sns.set()

#----------- Global variables -----------#
nsamp = 1000
A=1
values=np.zeros(nsamp)
trials=list(range(nsamp))
sigma = 1

moose = np.zeros(nsamp)
rs = np.zeros(nsamp)
thetas = np.zeros(nsamp)

#---------- Experiment Definition--------#
def mean_uniform(n, m): # n = samples, m = nb of exp
    out = np.zeros(m) # index 0 is mean, 1 is std
    for i in range(m):
        result_uni = UNI(n)
        out[i] = result_uni[0]
        
    return out

def mean_linear(n, m):
    out = np.zeros(m) # index 0 is mean, 1 is std
    for i in range(m):
        result_uni = LIN(n)
        out[i] = result_uni[0]
        
    return out

def mean_gaussian(n, m):
    out = np.zeros(m) # index 0 is mean, 1 is std
    for i in range(m):
        result_uni = GAU(n)
        out[i] = result_uni[0]
        
    return out

stats_uni = np.zeros((2, 1000))
stats_lin = np.zeros((2, 1000))
stats_gau = np.zeros((2, 1000))


# Showing how the means and stds vary as n increases for the uniform model
r.seed(4)

for i in range(1000): 
    curr_uni = UNI(5*i+1)
    stats_uni[0][i], stats_uni[1][i] = curr_uni[0], curr_uni[1]
    
    curr_lin = LIN(i+1)
    stats_lin[0][i], stats_lin[1][i] = curr_lin[0], curr_lin[1]
    
    curr_gau = GAU(i+1)
    stats_gau[0][i], stats_gau[1][i] = curr_gau[0], curr_gau[1]
    
lin_xs = np.linspace(1, 5000, 1000)

#-- Plotting mu and std as n increases --#

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.scatter(lin_xs, stats_uni[0], 0.75)
ax1.set_title("$\mu$ as a function of $n$ following a uniform distribution")
ax2.scatter(lin_xs, stats_uni[1], 0.75, color='orange')
ax2.set_title("$\sigma$ as a function of $n$ following a uniform distribution")
fig.tight_layout()
plt.xlabel('Sample size (n) of experiment')
plt.savefig('../run/figures/'+'uniformIncreasingSampleSize.png')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.scatter(lin_xs, stats_lin[0], 0.75)
ax1.set_title("$\mu$ as a function of $n$ following a linear distribution")
ax2.scatter(lin_xs, stats_lin[1], 0.75, color='orange')
ax2.set_title("$\sigma$ as a function of $n$ following a linear distribution")
fig.tight_layout()
plt.xlabel('Sample size (n) of experiment')
plt.savefig('../run/figures/'+'linearIncreasingSampleSize.png')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.scatter(lin_xs, stats_gau[0], 0.75)
ax1.set_title("$\mu$ as a function of $n$ following a gaussian distribution")
ax2.scatter(lin_xs, stats_gau[1], 0.75, color='orange')
ax2.set_title("$\sigma$ as a function of $n$ following a gaussian distribution")
fig.tight_layout()
plt.xlabel('Sample size (n) of experiment')
plt.savefig('../run/figures/'+'gaussianIncreasingSampleSize.png')
plt.show()

#----------- Plotting mean histograms -----------#

expUni1 = mean_uniform(25, 1000) #output of len 1000, representing the means of 1000 exp with n=25
expUni2 = mean_uniform(100, 1000)
expUni3 = mean_uniform(500, 1000)
expUni4 = mean_uniform(1000, 1000)

expLin1 = mean_linear(25, 1000) #output of len 1000, representing the means of 1000 exp with n=25
expLin2 = mean_linear(100, 1000)
expLin3 = mean_linear(500, 1000)
expLin4 = mean_linear(1000, 1000)

expGau1 = mean_gaussian(25, 1000) #output of len 1000, representing the means of 1000 exp with n=25
expGau2 = mean_gaussian(100, 1000)
expGau3 = mean_gaussian(500, 1000)
expGau4 = mean_gaussian(1000, 1000)

#-----------   Uniform Distribution   -----------#
# Define the min and max of the x axis
xmin = min(expUni1.min(), expUni2.min(), expUni3.min(), expUni4.min())
xmax = max(expUni1.max(), expUni2.max(), expUni3.max(), expUni4.max())

hist_uni = plt.figure()
plt.subplot(2, 2, 1)
plt.hist(expUni1)
plt.ylabel('Count')
plt.title('(a)')
# Set the x axis limits
plt.xlim(xmin,xmax)

plt.subplot(2, 2, 2)
plt.hist(expUni2)
plt.title('(b)')
# Set the x axis limits
plt.xlim(xmin,xmax)

plt.subplot(2, 2, 3)
plt.hist(expUni3)
plt.ylabel('Count')
plt.xlabel('values')
plt.title('(c)')
# Set the x axis limits
plt.xlim(xmin,xmax)

plt.subplot(2, 2, 4)
plt.hist(expUni4)
plt.xlabel('values')
plt.title('(d)')
# Set the x axis limits
plt.xlim(xmin,xmax)
        
plt.tight_layout()
    
plt.savefig('../run/figures/'+'meanHistogramUniPlot.png')


#-----------   Uniform Distribution   -----------#
# Define the min and max of the x axis
xmin = min(expLin1.min(), expLin2.min(), expLin3.min(), expLin4.min())
xmax = max(expLin1.max(), expLin2.max(), expLin3.max(), expLin4.max())

hist_uni = plt.figure()
plt.subplot(2, 2, 1)
plt.hist(expLin1)
plt.ylabel('Count')
plt.title('(a)')
# Set the x axis limits
plt.xlim(xmin,xmax)

plt.subplot(2, 2, 2)
plt.hist(expLin2)
plt.title('(b)')
# Set the x axis limits
plt.xlim(xmin,xmax)

plt.subplot(2, 2, 3)
plt.hist(expLin3)
plt.ylabel('Count')
plt.xlabel('values')
plt.title('(c)')
# Set the x axis limits
plt.xlim(xmin,xmax)

plt.subplot(2, 2, 4)
plt.hist(expLin4)
plt.xlabel('values')
plt.title('(d)')
# Set the x axis limits
plt.xlim(xmin,xmax)
        
plt.tight_layout()
    
plt.savefig('../run/figures/'+'meanHistogramLinPlot.png')

#-----------   Uniform Distribution   -----------#
# Define the min and max of the x axis
xmin = min(expGau1.min(), expGau2.min(), expGau3.min(), expGau4.min())
xmax = max(expGau1.max(), expGau2.max(), expGau3.max(), expGau4.max())

hist_uni = plt.figure()
plt.subplot(2, 2, 1)
plt.hist(expGau1)
plt.ylabel('Count')
plt.title('(a)')
# Set the x axis limits
plt.xlim(xmin,xmax)

plt.subplot(2, 2, 2)
plt.hist(expGau2)
plt.title('(b)')
# Set the x axis limits
plt.xlim(xmin,xmax)

plt.subplot(2, 2, 3)
plt.hist(expGau3)
plt.ylabel('Count')
plt.xlabel('values')
plt.title('(c)')
# Set the x axis limits
plt.xlim(xmin,xmax)

plt.subplot(2, 2, 4)
plt.hist(expGau4)
plt.xlabel('values')
plt.title('(d)')
# Set the x axis limits
plt.xlim(xmin,xmax)
        
plt.tight_layout()
    
plt.savefig('../run/figures/'+'meanHistogramGauPlot.png')
plt.show()