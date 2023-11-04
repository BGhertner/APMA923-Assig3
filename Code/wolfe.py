"""
Bisection for Wolfe conditions - Nick Huang, November 2023
"""

import numpy as np

def wolfe_bisection(f, g, xk, pk, gk, mu=1e-10, sigma=0.7, 
                    alpha = 0., t = 1., beta = np.inf, gmethod=0, obj_calls=0):
    """
    Args:
        f: function handle of function to be minimized
        g: function handle of gradient of f
        xk: input vector at current step
        pk: chosen descent direction
        gk: gradient of f at xk
        mu: hyperparmeter with 0 < mu < sigma  < 1
        sigma: (optional) hyperparameter with 0 < mu < sigma < 1
        alpha: (optional) parameter to be used in wolfe, can be left untouched
        t: current proposed stepsize, can be left untouched
        beta: parameter to be used in wolfe, can be left untouched.
        gmethod: Which grad method is used (this only matter for how to count funct. calls)
        obj_calls: current function calls count
    Returns: 
        float t, the step length
        obj_calls: number of times the objective function was called

    """
    obj_calls = 0
    xk = xk.reshape(-1,1)
    d = xk.shape[0]

    gxk1 = g(xk + t * pk)  # Create gradient of next vector
    if gmethod == 0: obj_calls += 1
    elif gmethod == 1: obj_calls += d

    obj_calls += 2 #2 calls to evaluate if
    if f(xk + t * pk) > f(xk) + mu * t * np.dot(gk.T, pk):  # Check armijo
        return wolfe_bisection(f, g, xk, pk, gk, mu, sigma, alpha, 
                               t=(alpha+t)/2, beta=t, gmethod=gmethod, obj_calls=obj_calls)  # If yes, reset
    elif np.dot(gxk1.T, pk) < sigma * np.dot(gk.T, pk):  # Wolfe
        if beta == np.inf:
            # If infinite beta, reset t = 2 * alpha
            return wolfe_bisection(f, g, xk, pk, gk, mu, sigma, alpha=t, 
                                   t=2*t, beta=beta, gmethod=gmethod, obj_calls=obj_calls)
        else:
            # Otherwise, reset t = (alpha + beta)/2
            return wolfe_bisection(f, g, xk, pk, gk, mu, sigma, alpha=t, 
                                   t=(t + beta)/2, beta=beta, gmethod=gmethod, obj_calls=obj_calls)
    else:
        return t, obj_calls  # We have our step


def soft_abs(x, eps=0.0001):
    """
    Numerically stable absolute value in terms of derivatives.

    Args:
        x: The point at which to evaluate the absolute value.
        eps: Optional small positive float, smaller the better

    Returns: float, soft abs of x

    """
    return np.sqrt(x**2 + eps**2)

def strong_wolfe_bisection(f, g, xk, pk, gk, mu=0.25, sigma=0.75, alpha = 0., t = 1., beta = np.inf):
    """
    DOESN'T WORK BUT YOU CAN PLAY WITH IT IF YOU WANT

    Args:
        f: function handle of function to be minimized
        g: function handle of gradient of f
        xk: input vector at current step
        pk: chosen descent direction
        gk: gradient of f at xk
        mu: hyperparmeter with 0 < mu < sigma  < 1
        sigma: (optional) hyperparameter with 0 < mu < sigma < 1
        alpha: (optional) parameter to be used in wolfe, can be left untouched
        t: current proposed stepsize, can be left untouched
        beta: parameter to be used in wolfe, can be left untouched.
    Returns: float, the step length

    """
    gxk1 = g(xk + t * pk)  # Create gradient of next vector

    if f(xk + t * pk) > f(xk) + mu * t * np.dot(gk, pk):  # Armjio
        wolfe_bisection(f, xk, pk, gk, mu, sigma, alpha, t=(alpha+t)/2, beta=t)  # If yes, reset
    elif soft_abs(np.dot(gxk1, pk)) > soft_abs(sigma * np.dot(gk, pk)):  # Uses soft abs to avoid numerical problems
        if beta == np.inf:
            # If infinite beta, reset t = 2 * alpha
            wolfe_bisection(f, g, xk, pk, gk, mu, sigma, alpha=2*t, t=2*alpha, beta=beta)
        else:
            # Otherwise, reset t = (alpha + beta)/2
            wolfe_bisection(f, g, xk, pk, gk, mu, sigma, alpha=2 * t, t=(alpha + beta)/2, beta=beta)
    else:
        return t  # We have our step