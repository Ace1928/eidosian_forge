import sys
def xaxpy_e(x, y, alpha=1.0):
    xaxpy(x[0], y[0], alpha=alpha)
    y[1] += alpha * x[1]