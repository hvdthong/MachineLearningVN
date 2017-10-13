import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

np.random.seed(21)
means = [[2, 2], [4, 1]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X1[-1, :] = [2.7, 2]
X = np.concatenate((X0.T, X1.T), axis=1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
print X.shape, y.shape

C = 100
# build K
V = np.concatenate((X0.T, -X1.T), axis=1)
K = matrix(V.T.dot(V))

p = matrix(-np.ones((2 * N, 1)))
# build A, b, G, h
G = matrix(np.vstack((-np.eye(2 * N), np.eye(2 * N))))

h = matrix(np.vstack((np.zeros((2 * N, 1)), C * np.ones((2 * N, 1)))))
A = matrix(y.reshape((-1, 2 * N)))
b = matrix(np.zeros((1, 1)))
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
print('lambda = \n', l.T)

S = np.where(l > 1e-5)[0]  # support set
S2 = np.where(l < .99 * C)[0]

M = [val for val in S if val in S2]  # intersection of two lists
XT = X.T  # we need each column to be one data point in this alg
VS = V[:, S]
lS = l[S]
yM = y.T[M, :]
XM = XT[M, :]

w_dual = VS.dot(lS).reshape(-1, 1)
b_dual = np.mean(yM.T - w_dual.T.dot(XM.T))
print(w_dual.T, b_dual)
