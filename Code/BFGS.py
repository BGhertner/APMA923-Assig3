"""
BFGS - Benjamin Ghertner Oct 2023
"""

import numpy as np
import scipy.linalg as sla


def BFGS(x, f=None, g=None, H=None, 
         get_step=None, eps=1e-4, Xmax=10, 
         kmax=100, verbose=False, gmethod=0, h=None):
    """
    BFGS - Benjamin Ghertner Oct 2023

    Inputs:

    x: (1D array) inital start point.

    f: default=None, function handle for a function which takes an x position as an 
        input and outputs the value of the objective function at x.

    g: (optional) default=None, function handle for a function which takes an x position as an
        input and outputs a d X 1 vector cooresponding to the gradient of the objective function
        at the point x.

    H: (optional) default=None, d X d numpy array, initial approximation for the inverse hessian 
        approximation "H" this matrix needs to be symetric if it is not positive definite then a
        mu will be found so that H + mu I > 0 and the need H = H + mu I will be used instead.

    get_step: (required) default=None, function handle which takes the below inputs and
                         returns the step-length for updating x

        f = function handle which takes a position vector, x, and returns the objective function at
            that point.

        xk = d X 1 - numpy vector, curent x position of the algorithm

        pk = d X 1 - numpy vector, search direction at current iteration

        gk = d X 1 - numpy vector, gradient of the objective function at xk

        alpha1 = scalar - step size for the last iteration
        
        alpha2 = scalar - step size for 2 iterations ago

    eps: (optional) default=1e-4, scalar - tolerance for stopping criteria 
        min(||gk||, ||xk - xk-1||) <= eps.

    Xmax: (optional) default=10, scalar - maximum distance from the start point for optimization
        search. Algorithm stops if ||xk - x0|| > Xmax

    kmax: (optional) default=100, scalar - maximum number of iterations of search algorithm.

    verbose: (optional) default=False, boolean - if Ture print xk and gk at each iteration.

    gmethod: (optional) default=1, integer - option for gradient calculation,

        0 - Use a supplied function for the gradient. g must be provided in this case.

        1 - Complex differentiation is used to create an approximation of g.

    h: (optional) the "h" parameter in complex differentiation. If not provided and gmethod = 1 
        then h = eps will be used.

    Returns:

    xk: d X 1 numpy vector - Final x point of the algorithm.

    grads: list - list of the norms of the gradients at each iteration

    obj_calls: scalar - number of calls to the objective function
    """

    #Initialize the objective calls count and list of gradient norms
    obj_calls = 0
    grads = []

    #Start off x1 and x0 as the initial x point
    x1 = x0 = x.reshape(-1,1) #reshape to column vector if not already
    d = x0.shape[0] #columns of x are number of dimensions

    #if gmethod is not 0 then create a lambda function for g to be used
    if gmethod == 1: #complex differentiation
        if h is None:
            h = eps
        #Define a complex diff. function
        def get_g(xk):
            gk = np.zeros_like(xk)
            for i in range(xk.size):
                inp = np.zeros_like(xk).astype('complex128')
                inp += xk
                inp[i] += 1j*h
                gk[i] = (f(inp)/h).imag
            return gk
        #Set g to the complex diff function
        g = get_g

    #Start off g1 and g2 as the grad at the initial x point
    g1 = g0 = g(x0).reshape(-1,1)

    #Count function calls
    if gmethod == 0: obj_calls += 1
    elif gmethod == 1: obj_calls += d

    #ID matrix for convience
    I = np.eye(d,d)
    
    #Check that the initial H is SPD
    eigvals = sla.eigvals(H)
    #Fix if not...
    if np.any(eigvals <= 0 ): H = H - np.min(eigvals)*1.1*I

    #initialize iteration count (k)
    k = 0

    #Initilization of step "previous" step sizes variables
    alpha1 = 1

    #BFGS main loop
    #Stop if xk is outside Xmax distance from initial point
    #Stop if number of iterations (k) > max iterations (kmax)
    #Stop if ||grad(xk)|| or ||xk-1 - xk|| < stopping tolerance (eps)
    while (sla.norm(x1 - x.reshape(-1,1), 2) < Xmax \
           and k < kmax \
           and np.min([sla.norm(g1,2), sla.norm(x0 - x1, 2)]) > eps) \
           or k < 2: #Run for at least two iterations.
        
        #Numbered steps following algorithm 4.1 p.25 on notes
        #Part 4 - Second order methods, Lecture 3 - Practical Implementation

        #1
        x0 = x1
        g0 = g1

        #Method "gets stuck" reset H as I
        if sla.norm(x0 - x1) < 1e-4: H = np.eye(d,d)

        #2
        p = -H@g0
        
        #3
        alpha1, f_calls = get_step(f, g, x0, p, g0)
        obj_calls += f_calls

        #4
        s = alpha1*p

        #5
        x1 = x0 + s

        #6
        g1 = g(x1).reshape(-1,1)
        if gmethod == 0: obj_calls += 1
        elif gmethod == 1: obj_calls += d

        #save ||gk|| to return history of grad sizes
        grads.append(sla.norm(g1, 2))

        #7
        y = g1 - g0

        #8
        rho = 1/y.T@s

        #9
        H = (I - rho*s@y.T)@H@(I - rho*y@s.T) + rho*s@s.T

        #10
        k += 1

        if verbose: print(f'x = {x1}, g = {g1}')


    return x1, grads, obj_calls
