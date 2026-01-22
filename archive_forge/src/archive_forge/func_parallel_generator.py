import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
def parallel_generator(generator):
    """Decorator for broadcasting yields from master to slaves using MPI.

    Disable by passing parallel=False to the function.  For a method,
    you can also disable the parallel behavior by giving the instance
    a self.serial = True.
    """

    @functools.wraps(generator)
    def new_generator(*args, **kwargs):
        if world.size == 1 or (args and getattr(args[0], 'serial', False)) or (not kwargs.pop('parallel', True)):
            for result in generator(*args, **kwargs):
                yield result
            return
        if world.rank == 0:
            try:
                for result in generator(*args, **kwargs):
                    broadcast((None, result))
                    yield result
            except Exception as ex:
                broadcast((ex, None))
                raise ex
            broadcast((None, None))
        else:
            ex2, result = broadcast((None, None))
            if ex2 is not None:
                raise ex2
            while result is not None:
                yield result
                ex2, result = broadcast((None, None))
                if ex2 is not None:
                    raise ex2
    return new_generator