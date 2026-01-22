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
def user_prop_pop(ctx, cb, num_scopes):
    prop = _prop_closures.get(ctx)
    prop.cb = cb
    prop.pop(num_scopes)