import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def cxxid(name):
    from pythran.tables import cxx_keywords
    return name + '_' * (name in cxx_keywords)