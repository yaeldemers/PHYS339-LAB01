# 7.2 - GAUSSIAN MODEL


#------------Import statements-----------#
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import random as r
import seaborn as sns
sns.set()


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

#q2data = gaussian_model(1000)
xs, ys = gaussian_model(1000)

plt.plot(xs, ys, 'o', markersize=3)
plt.title('1000 data points - gaussian distributed on x and y')
plt.show()

# 1000 data points generated from gaussian distribution on the dependent variable
uniform_xs = np.arange(0, 1000, 1)
plt.plot(uniform_xs, ys, 'o', markersize=3)
plt.title(' 1000 data points - gaussian distributed')
plt.show()


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

def GAU(n): #used for data generation of section 7.3
    values = gaussian_model(n)[1]
    return([np.mean(values), np.std(values)])

    