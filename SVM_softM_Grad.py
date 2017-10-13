import numpy as np


def cost(w):
    u = w.T.dot(Z)  # as in (23)
    return (np.sum(np.maximum(0, 1 - u)) + .5 * lam * np.sum(w * w)) - .5 * lam * w[-1] * w[-1]  # no bias


def grad(w):
    u = w.T.dot(Z)  # as in (23)
    H = np.where(u < 1)[1]
    ZS = Z[:, H]
    g = (-np.sum(ZS, axis=1, keepdims=True) + lam * w)
    g[-1] -= lam * w[-1]  # no weight decay on bias
    return g


def num_grad(w):
    g = np.zeros_like(w)
    for i in xrange(len(w)):
        wp = w.copy()
        wm = w.copy()
        wp[i] += eps
        wm[i] -= eps
        g[i] = (cost(wp) - cost(wm)) / (2 * eps)
    return g


def grad_descent(w0, eta):
    w = w0
    it = 0
    while it < 100000:
        it = it + 1
        g = grad(w)
        w -= eta * g
        if (it % 10000) == 1:
            print('iter %d' % it + ' cost: %f' % cost(w))
        if np.linalg.norm(g) < 1e-5:
            break
    return w


np.random.seed(21)
means = [[2, 2], [4, 1]]
cov = [[.3, .2], [.2, .3]]
N, C = 10, 100
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X1[-1, :] = [2.7, 2]
X = np.concatenate((X0.T, X1.T), axis=1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
print X.shape, y.shape

X0_bar = np.vstack((X0.T, np.ones((1, N))))  # extended data
X1_bar = np.vstack((X1.T, np.ones((1, N))))  # extended data
Z = np.hstack((X0_bar, - X1_bar))  # as in (22)
lam = 1. / C
eps = 1e-6

w0 = np.random.randn(X0_bar.shape[0], 1)
w = grad_descent(w0, 0.001)
w_hinge = w[:-1].reshape(-1, 1)
b_hinge = w[-1]
print(w_hinge.T, b_hinge)
