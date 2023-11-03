import numpy as np
import matplotlib.pyplot as plt
from NN_BP import NN_BP
from EvalNetwork import EvalNetwork

def BuildDSData(n):
    # function to build some fake data to classify.
    tx = -1 + 2 * np.random.rand(n)
    ty = -1 + 2 * np.random.rand(n)

    f = lambda x, y: -x / 4 + y / 6 + 0.035 * np.cos(2 * np.pi * x) + 0.15 * x**2 - 0.25 * y**2 + 0.5 * np.exp(-0.02 * y)

    ym = np.mean(f(tx, ty))
    y = np.zeros(n)
    for i in range(n):
        y[i] = f(tx[i], ty[i]) > ym

    t = np.column_stack((tx, ty))
    return t, y

# the main script
n = 200
t, y = BuildDSData(n)
W, b, _ = NN_BP(t.T, y.reshape(1, -1), [2, 3, 2])

xx, yy = np.meshgrid(np.linspace(-1, 1, 201), np.linspace(-1, 1, 201))
zz = np.zeros_like(xx)

for i in range(xx.size):
    zz.ravel()[i] = EvalNetwork(np.array([xx.ravel()[i], yy.ravel()[i]]), W, b)

plt.contour(xx, yy, zz, [0.5], colors='k', linewidths=3)
plt.plot(t[y > 0.5, 0], t[y > 0.5, 1], 'rx')
plt.plot(t[y < 0.5, 0], t[y < 0.5, 1], 'bs')

for i in range(4):
    W, b, _ = NN_BP(t.T, y.reshape(1, -1), [2, 3, 2], W, b)

for i in range(xx.size):
    zz.ravel()[i] = EvalNetwork(np.array([xx.ravel()[i], yy.ravel()[i]]), W, b)

plt.contour(xx, yy, zz, [0.5], colors='g', linewidths=3)
plt.show()
