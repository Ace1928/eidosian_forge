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
def declare_core(self, name, rec_name, *args):
    if z3_debug():
        _z3_assert(isinstance(name, str), 'String expected')
        _z3_assert(isinstance(rec_name, str), 'String expected')
        _z3_assert(all([_valid_accessor(a) for a in args]), 'Valid list of accessors expected. An accessor is a pair of the form (String, Datatype|Sort)')
    self.constructors.append((name, rec_name, args))