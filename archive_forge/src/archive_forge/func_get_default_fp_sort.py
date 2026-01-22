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
def get_default_fp_sort(ctx=None):
    return FPSort(_dflt_fpsort_ebits, _dflt_fpsort_sbits, ctx)