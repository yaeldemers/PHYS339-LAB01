# UNIFORM MODEL FROM TUTORIAL

import random as r
import numpy as np

def UNI(nsamp):
    values=np.zeros(nsamp)
    trials=list(range(nsamp))
    for i in trials:
        values[i]=r.random()

    return([np.mean(values), np.std(values)])
