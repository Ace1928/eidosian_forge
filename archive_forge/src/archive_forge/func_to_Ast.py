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
def to_Ast(ptr):
    ast = Ast(ptr)
    super(ctypes.c_void_p, ast).__init__(ptr)
    return ast