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
def mean_uniform(n, m): #n=nb per exp, m=nb of exp
    out = np.zeros(m) # index 0 is mean, 1 is std
    for i in range(m):
        result_uni = UNI(n)
        out[i] = result_uni[0]
        
    return out

def stats_linear(n):
    return 0

def stats_gaussian(n):
    return 0

x = [0,1,2,3,4,5,6,7,8,9]

exp1 = mean_uniform(25, 1000) #output of len 1000, representing the means of 1000 exp with n=25
exp2 = mean_uniform(100, 1000)
exp3 = mean_uniform(500, 1000)
exp4 = mean_uniform(1000, 1000)


# Tentative 1
m1 = np.mean(exp1)
m2 = np.mean(exp2)
m3 = np.mean(exp3)
m4 = np.mean(exp4)


plt.scatter([25, 100, 500, 1000], [m1, m2, m3, m4])
plt.show()

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

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.scatter(lin_xs, stats_lin[0], 0.75)
ax1.set_title("$\mu$ as a function of $n$ following a linear distribution")
ax2.scatter(lin_xs, stats_lin[1], 0.75, color='orange')
ax2.set_title("$\sigma$ as a function of $n$ following a linear distribution")
fig.tight_layout()

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.scatter(lin_xs, stats_gau[0], 0.75)
ax1.set_title("$\mu$ as a function of $n$ following a gaussian distribution")
ax2.scatter(lin_xs, stats_gau[1], 0.75, color='orange')
ax2.set_title("$\sigma$ as a function of $n$ following a gaussian distribution")
fig.tight_layout()

