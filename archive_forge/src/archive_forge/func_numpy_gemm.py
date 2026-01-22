import numpy
import numpy.random
from .py import gemm, einsum
from timeit import default_timer as timer
def numpy_gemm(X, W, n=1000):
    nO, nI = W.shape
    batch_size = X.shape[0]
    total = 0.0
    y = numpy.zeros((batch_size, nO), dtype='f')
    for i in range(n):
        numpy.dot(X, W, out=y)
        total += y.sum()
        y.fill(0)
    print('Total:', total)