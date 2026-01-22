import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
def runtime_build_excinfo_struct(static_exc, exc_args):
    exc, static_args, locinfo = cloudpickle.loads(static_exc)
    real_args = []
    exc_args_iter = iter(exc_args)
    for arg in static_args:
        if isinstance(arg, ir.Value):
            real_args.append(next(exc_args_iter))
        else:
            real_args.append(arg)
    return (exc, tuple(real_args), locinfo)