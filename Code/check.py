import numpy as np

from CG import *
from backtracking import *
from wolfe import *
from Adam import *

def sim(x):
    return -1*np.exp(-1*((x[0]*x[1]-1.5)**2 + (x[1]-1.5)**2))

def der_sim(x):
    f = sim(x)
    f1 = -2*(x[0]*x[1]-1.5)*x[1]
    f2 = -2*(x[0]*x[1]-1.5)*x[0] -2*(x[1]-1.5)
    return np.array([f*f1, f*f2])

optim = AdaM(np.array([2, 2]), sim, der_sim, wolfe_bisection, gamma1=0.9, gamma2=0.9, kmax=200, verbose=False)
print(optim[0].T)