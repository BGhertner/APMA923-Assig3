import numpy as np

def EvalNetwork(x, WW, bb):
    nn = len(WW)
    a = x
    for l in range(1, nn):
        a = activate(a, WW[l], bb[l])
    return a

def activate(x, W, b):
    x = x.reshape(-1,1)
    b = b.reshape(-1,1)
    return 1 / (1 + np.exp(-(W@x + b)))

# Example usage:
# a = EvalNetwork(x, WW, bb)
