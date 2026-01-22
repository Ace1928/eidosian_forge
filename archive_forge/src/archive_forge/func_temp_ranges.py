import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
def temp_ranges(*args):
    return [gfun(args[0]) if callable(gfun) else gfun, hfun(args[0]) if callable(hfun) else hfun]