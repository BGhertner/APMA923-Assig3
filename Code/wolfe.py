"""
Bisection for Wolfe conditions - Nick Huang, November 2023
"""

import jax.numpy as np

def wolfe_bisection(f, g, xk, pk, gk, mu=0.25, sigma=0.75, alpha = 0., t = 1., beta = np.inf):
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
    Returns: float t, the step length

    """
    gxk1 = g(f, xk + t * pk)  # Create gradient of next vector

    if f(xk + t * pk) > f(xk) + mu * t * np.dot(gk, pk):  # Check armijo
        wolfe_bisection(f, xk, pk, gk, mu, sigma, alpha, t=(alpha+t)/2, beta=t)  # If yes, reset
    elif np.dot(gxk1, pk) > sigma * np.dot(gk, pk):  # Wolfe
        if beta == np.inf:
            # If infinite beta, reset t = 2 * alpha
            wolfe_bisection(f, xk, pk, gk, mu, sigma, gmethod, alpha=2*t, t=2*alpha, beta=beta)
        else:
            # Otherwise, reset t = (alpha + beta)/2
            wolfe_bisection(f, xk, pk, gk, mu, sigma, gmethod, alpha=2 * t, t=(alpha + beta)/2, beta=beta)
    else:
        return t  # We have our step


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
    gxk1 = gmethod(f, xk + t * pk)  # Create gradient of next vector

    if f(xk + t * pk) > f(xk) + mu * t * np.dot(gk, pk):  # Armjio
        wolfe_bisection(f, xk, pk, gk, mu, sigma, alpha, t=(alpha+t)/2, beta=t)  # If yes, reset
    elif soft_abs(np.dot(gxk1, pk)) > soft_abs(sigma * np.dot(gk, pk)):  # Uses soft abs to avoid numerical problems
        if beta == np.inf:
            # If infinite beta, reset t = 2 * alpha
            wolfe_bisection(f, xk, pk, gk, mu, sigma, gmethod, alpha=2*t, t=2*alpha, beta=beta)
        else:
            # Otherwise, reset t = (alpha + beta)/2
            wolfe_bisection(f, xk, pk, gk, mu, sigma, gmethod, alpha=2 * t, t=(alpha + beta)/2, beta=beta)
    else:
        return t  # We have our step