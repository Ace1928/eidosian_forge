import common_z3 as CM_Z3
import ctypes
from .z3 import *
def mk_var(name, vsort):
    if vsort.kind() == Z3_INT_SORT:
        v = Int(name)
    elif vsort.kind() == Z3_REAL_SORT:
        v = Real(name)
    elif vsort.kind() == Z3_BOOL_SORT:
        v = Bool(name)
    elif vsort.kind() == Z3_DATATYPE_SORT:
        v = Const(name, vsort)
    else:
        raise TypeError('Cannot handle this sort (s: %sid: %d)' % (vsort, vsort.kind()))
    return v