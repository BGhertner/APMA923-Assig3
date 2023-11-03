import numpy as np

def NN_BP(T, Y, dl, W=None, b=None):
    dIn, N = T.shape
    if dIn > N:
        T = T.T
        dIn, N = T.shape
    dOut, m = Y.shape
    if dOut > m:
        Y = Y.T
        dOut, m = Y.shape

    if N != m:
        print('Data error mismatch, try again!')
        return

    dvec = np.hstack((dIn, dl, dOut))
    L = len(dvec)

    # Initialize weights and biases
    np.random.seed(5000)
    TDim = 0
    if W is None or b is None:
        W = [None]
        b = [None]
        for i in range(1, L):
            W.append(np.random.rand(dvec[i], dvec[i - 1]))
            b.append(np.random.rand(dvec[i], 1))
            TDim += dvec[i] * (1 + dvec[i - 1])
    else:
        for i in range(1, L):
            TDim += dvec[i] * (1 + dvec[i - 1])
    delta = b.copy()
    a = b.copy()
    W0 = W.copy()
    b0 = b.copy()

    # Parameters
    eta = 0.1  # learning rate
    Niter = int(1e5)  # number of SG iterations
    dtheta = np.zeros(int(np.ceil(Niter / N) + 10))
    kk = 0

    def activate(x, W, b):
        x = x.reshape(-1,1)
        b = b.reshape(-1,1)
        return 1 / (1 + np.exp(-(W@x + b)))

    # Forward and Back propagate
    for counter in range(Niter):
        k = np.random.randint(0, N)  # choose a training point at random
        x = T[:, k]
        y = Y[:, k]
        # Forward pass
        a[0] = x
        for l in range(1, L):
            a[l] = activate(a[l - 1], W[l], b[l])
        # Backward pass
        delta[L-1] = a[L-1] * (1 - a[L-1]) * (a[L-1] - y)
        for l in range(L - 2, -1, -1):
            delta[l] = a[l] * (1 - a[l]) * np.dot(W[l + 1].T, delta[l + 1])
        # Gradient step
        wnorm = 0
        bnorm = 0
        for l in range(1, L):
            dW = eta * np.outer(delta[l], a[l - 1])
            wnorm += np.linalg.norm(dW)
            W[l] -= dW
            db = eta * delta[l]
            bnorm += np.linalg.norm(db)
            b[l] -= db
        # Monitor progress
        if counter % N == 0:
            wnorm = 0
            bnorm = 0
            for l in range(1, L):
                wnorm += np.linalg.norm(W[l] - W0[l])
                W0[l] = W[l]
                bnorm += np.linalg.norm(b[l] - b0[l])
                b0[l] = b[l]
                #dtheta[kk] = (wnorm + bnorm) / 2 / TDim
                kk += 1

    #dtheta = dtheta[:kk]

    return W, b, dtheta

# Example usage:
# W, b, dtheta = NN_BP(T, Y, dl)
