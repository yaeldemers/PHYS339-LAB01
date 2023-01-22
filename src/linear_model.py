# PART 7.1 - LINEAR RANDOM NUMBERS


#------------Import statements-----------#
import matplotlib.pyplot as plt
import numpy as np
import random as r
import seaborn as sns
sns.set()


#------------Global variables-----------#
A=2 # Obtained by integrating PDF over 0-1
nsamp = 1000
values=np.zeros(nsamp)
trials=list(range(nsamp))
count_results= np.zeros((10,20))


#----------- Linear PDF & CDF ----------#
def linearModelPDF(A, x):
    x = np.array(x)
    return A*x

# CDF is obtained by integrating PDF from -inf to inf
def linearModelCDF(A, x): 
    x = np.array(x)
    return A*(x**2)/2

#--------- Apply reverse rule ---------#
def reverse(data):
    for i in range(len(data)):
        data[i] = np.sqrt(2*r.random()/A)
 
#--------- Used in mean dist ----------#
def LIN(n):
    data =np.zeros(n)
    reverse(data)
    return [np.mean(data), np.std(data)]

#----- Core of exp to be repeated -----#
def experiment_linear(exp_name, save, stats=None):    
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

    if save:
        # Find bin widths
        binWidthHist=np.diff(binEdges1)

        # Random values plot
        norm = nsamp*binWidthHist[0]
        f1 = plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(trials, values, 'o', markersize=0.75)
        plt.xlabel('trials')
        plt.ylabel('values')
        plt.title('(a)')
        
        # PDF plot
        plt.subplot(2, 2, 2)
        plt.plot(binCenterHist, histValues1, 'o', markersize=3)
        plt.plot(binCenterHist, norm*linearModelPDF(A, binCenterHist), 'k', linewidth=0.75)
        if stats is not None:
            plt.errorbar(binCenterHist, histValues1, yerr=stats, fmt="o", markersize=0, linewidth=1)
            plt.xlabel('values')
            plt.ylabel('counts')
        plt.title('(b)')

        # CDF plot
        plt.subplot(2, 2, 3)
        plt.plot(binCenterCumHist, cumHistValues, 'o', markersize=3)
        plt.plot(binCenterCumHist, linearModelCDF(A, binCenterCumHist), 'k', linewidth=0.75)
        plt.xlabel('value')
        plt.ylabel('cumulative counts')
        plt.title('(c)')

        plt.tight_layout()
    
        plt.savefig('../run/figures/'+exp_name+'Plot.png')

    # These commands output the histogram and cumulative histogram results to csv files
    # for plotting
    np.savetxt('../run/data/'+exp_name+'Hist.csv',
    np.transpose([binCenterHist, histValues1]), delimiter = ",")
    np.savetxt('../run/data/'+exp_name+'CumHist.csv',
    np.transpose([binCenterCumHist, cumHistValues]), delimiter = ",")
    
    return histValues1


#--------- Repeating the experiment ---------#
for i in range(10):
    r.seed(i) # for experiment reproducibility
    exp_name = 'trial'+str(i)
    count_results[i] = experiment_linear(exp_name, save=False)


means, stds = np.mean(count_results, axis=0), np.std(count_results, axis=0)        

#---- Experiment to compare to prev data ----#
r.seed(11)
out = experiment_linear('linearModel', save=True, stats=stds)
