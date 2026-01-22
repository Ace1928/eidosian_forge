import os
from ctypes.macholib.framework import framework_info
from ctypes.macholib.dylib import dylib_info
from itertools import *
def dyld_env(env, var):
    if env is None:
        env = os.environ
    rval = env.get(var)
    if rval is None:
        return []
    return rval.split(':')