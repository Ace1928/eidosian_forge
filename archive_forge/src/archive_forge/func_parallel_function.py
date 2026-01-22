import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
def parallel_function(func):
    """Decorator for broadcasting from master to slaves using MPI.

    Disable by passing parallel=False to the function.  For a method,
    you can also disable the parallel behavior by giving the instance
    a self.serial = True.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if world.size == 1 or (args and getattr(args[0], 'serial', False)) or (not kwargs.pop('parallel', True)):
            return func(*args, **kwargs)
        ex = None
        result = None
        if world.rank == 0:
            try:
                result = func(*args, **kwargs)
            except Exception as x:
                ex = x
        ex, result = broadcast((ex, result))
        if ex is not None:
            raise ex
        return result
    return new_func