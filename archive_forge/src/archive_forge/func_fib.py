import dill
from functools import partial
import warnings
def fib(n):
    assert n >= 0
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)