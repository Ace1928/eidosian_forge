from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def mk_not(a):
    if is_not(a):
        return a.arg(0)
    else:
        return Not(a)