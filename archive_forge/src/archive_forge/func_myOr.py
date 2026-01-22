import common_z3 as CM_Z3
import ctypes
from .z3 import *
def myOr(*L):
    return myBinOp(Z3_OP_OR, *L)