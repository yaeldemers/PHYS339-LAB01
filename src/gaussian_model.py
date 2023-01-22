# 7.2 - GAUSSIAN MODEL


#------------ Import statements ------------#
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import random as r
import seaborn as sns
sns.set()

r.seed(1011)

def gaussian_model(n, A=1, sigma=1):
    #------------ Mus and Thetas -----------#
    values=np.zeros(n)

    moose = np.zeros(n)
    rs = np.zeros(n)
    thetas = np.zeros(n)

    #------------ Mus and Thetas -----------#
    for i in range(len(values)):
        moose[i] = -(1/A) * np.log(1 - r.random())
        thetas[i] = r.random()*2*np.pi

    rs = np.sqrt(2*moose*sigma**2)

    ys = rs*np.sin(thetas)

    xs = rs*np.cos(thetas)
    
    return [xs, ys]

def GAU(n): # Used for data generation of section 7.3
    values = gaussian_model(n)[1]
    return([np.mean(values), np.std(values)])

def get_percentage_within_std(data, mu, std): # Used in results / Discussion
    one_std = 0
    two_std = 0
    for x in data:
        if abs(x-mu) <= std:
            one_std += 1
        if abs(x-mu) <= 2*std:
            two_std += 1
    return [one_std/len(data), two_std/len(data)]   

#-------------- Generate Data -------------#
xs, ys = gaussian_model(1000)

plt.plot(xs, ys, 'o', markersize=3)
plt.ylabel('Value')
plt.xlabel('Trial')
plt.savefig('../run/figures/'+'doubleGaussianPlot.png')
plt.show()

lin_xs = np.arange(0, 1000, 1)
plt.plot(lin_xs, ys, 'o', markersize=3)
plt.ylabel('Value')
plt.xlabel('Trial')
plt.savefig('../run/figures/'+'singleGaussianPlot.png')
plt.show()


#------------ Density Histogram -----------#
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

#--------------- Normal Curve --------------#
mu, std = 0, 1
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = stats.norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2, label='Norm(0,1)')
plt.ylabel('Density')
plt.xlabel('Value')
plt.legend()
plt.savefig('../run/figures/'+'densityHistPlot.png')
plt.show() 


#------------- Needed for report ------------#
in_one_std, in_two_std = get_percentage_within_std(ys, np.mean(ys), np.std(ys))

print("Within 1 std: "+str(in_one_std)) # 67.8%
print("Within 2 std: "+str(in_two_std)) # 95.7%