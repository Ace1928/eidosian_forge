import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
def refunc(x, *args):
    return func(x, *args).real