import numpy as np
from random import shuffle


# naive way to calculate loss and grad
def svm_loss_naive(W, X, y, reg):
    d, C = W.shape
    _, N = X.shape

    ## naive loss and grad
    loss = 0
    dW = np.zeros_like(W)
    for n in xrange(N):
        xn = X[:, n]
        score = W.T.dot(xn)
        for j in xrange(C):
            if j == y[n]:
                continue
            margin = 1 - score[y[n]] + score[j]
            if margin > 0:
                loss += margin
                dW[:, j] += xn
                dW[:, y[n]] -= xn

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)  # regularization

    dW /= N
    dW += reg * W  # gradient off regularization
    return loss, dW


# random, small data
# N, C, d = 10, 3, 5
# reg = .1
# W = np.random.randn(d, C)
# X = np.random.randn(d, N)
# y = np.random.randint(C, size=N)
#
# # sanity check
# print 'loss without regularization:', svm_loss_naive(W, X, y, 0)[0]
# print 'loss with regularization:', svm_loss_naive(W, X, y, .1)[0]


# more efficient way to compute loss and grad
def svm_loss_vectorized(W, X, y, reg):
    d, C = W.shape
    _, N = X.shape
    loss = 0
    dW = np.zeros_like(W)

    Z = W.T.dot(X)

    correct_class_score = np.choose(y, Z).reshape(N, 1).T
    margins = np.maximum(0, Z - correct_class_score + 1)
    margins[y, np.arange(margins.shape[1])] = 0
    loss = np.sum(margins, axis=(0, 1))
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    F = (margins > 0).astype(int)
    F[y, np.arange(F.shape[1])] = np.sum(-F, axis=0)
    dW = X.dot(F.T) / N + reg * W
    return loss, dW


N, C, d = 49000, 10, 3073
reg = .1
W = np.random.randn(d, C)
X = np.random.randn(d, N)
y = np.random.randint(C, size=N)

import time

# t1 = time.time()
# l1, dW1 = svm_loss_naive(W, X, y, reg)
# t2 = time.time()
# print 'Naive     : run time:', t2 - t1, '(s)'

t1 = time.time()
l2, dW2 = svm_loss_vectorized(W, X, y, reg)
t2 = time.time()
print 'Vectorized: run time:', t2 - t1, '(s)'


# print 'loss difference:', np.linalg.norm(l1 - l2)
# print 'gradient difference:', np.linalg.norm(dW1 - dW2)


# Mini-batch gradient descent
def multiclass_svm_GD(X, y, Winit, reg, lr=.1, batch_size=100, num_iters=1000, print_every=100):
    W = Winit
    loss_history = np.zeros((num_iters))
    for it in xrange(num_iters):
        # randomly pick a batch of X
        idx = np.random.choice(X.shape[1], batch_size)
        X_batch = X[:, idx]
        y_batch = y[idx]

        loss_history[it], dW = svm_loss_vectorized(W, X_batch, y_batch, reg)

        W -= lr * dW
        if it % print_every == 1:
            print 'it %d/%d, loss = %f' \
                  % (it, num_iters, loss_history[it])

    return W, loss_history

# N, C, d = 49000, 10, 3073
# reg = .1
# W = np.random.randn(d, C)
# X = np.random.randn(d, N)
# y = np.random.randint(C, size=N)
#
# W, loss_history = multiclass_svm_GD(X, y, W, reg)
# import matplotlib.pyplot as plt
#
# # plot loss as a function of iteration
# plt.plot(loss_history)
# plt.show()
