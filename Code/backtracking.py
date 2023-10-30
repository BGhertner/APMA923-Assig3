"""
Imporved backtracking - Benjamin Ghertner Oct 2023
"""

import numpy as np

def backtracking(f, xk, pk, gk, alpha1, alpha2, mu=1e-4, rho=1/2):
    """
    Imporved backtracking - Benjamin Ghertner Oct 2023

    Inputs:

    f: function handle for a function which takes an x position as an 
        input and outputs the value of the objective function at x.

    xk: (1D array) current optimization algorithm position.

    pk: (1D array) current step direction should be in a descent direction.

    gk: (1D array) gradient of f at xk.

    alpha1: (scalar) previous step size.

    alpha2: (scalar) step size for 2 previous steps ago.

    mu: (scalar) backtracking parameter.

    rho: (scalar) backtracking parameter.

    Returns:

    alpha: (scalar) stepsize selected.
    """
    if alpha2 == alpha1: alpha = alpha1/rho**2
    else: alpha = alpha1

    while f(xk + alpha*pk) > f(xk) + alpha*mu*np.dot(gk.T, pk):
        alpha = rho*alpha

    return alpha
