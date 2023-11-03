"""
Bisection for Wolfe conditions - Nick Huang, November 2023
"""

import numpy as np

def wolfe_bisection(f, xk, pk, gk, mu, sigma, gmethod, alpha = 0., t = 1., beta = np.inf):
    """
    Args:
        f: function handle to be minimized
        xk: input vector at current step
        pk: chosen descent direction
        gk: gradient of f at xk
        mu: hy.perparmeter with 0 < mu < sigma  < 1
        sigma: hyperparameter with 0 < mu < sigma < 1
        alpha: parameter to be used in wolfe, can be left untouched
        t: current proposed stepsize, can be left untouched
        beta: parameter to be used in wolfe, can be left untouched.
    Returns: float t, the step length

    """
    gxk1 = gmethod(f, xk + t * pk)

    if f(xk + t * pk) > f(xk) + mu * t * np.dot(gk, pk):
        wolfe_bisection(f, xk, pk, gk, mu, sigma, alpha, t=(alpha+t)/2, beta=t)
    elif np.dot(gxk1, pk) < sigma * np.dot(gk, pk):
        if beta == np.inf:
            wolfe_bisection(f, xk, pk, gk, mu, sigma, gmethod, alpha=2*t, t=2*alpha, beta=beta)
    else:
        return t