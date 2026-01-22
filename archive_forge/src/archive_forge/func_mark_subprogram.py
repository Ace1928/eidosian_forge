import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def mark_subprogram(self, function, qualname, argnames, argtypes, line):
    name = qualname
    argmap = dict(zip(argnames, argtypes))
    di_subp = self._add_subprogram(name=name, linkagename=function.name, line=line, function=function, argmap=argmap)
    function.set_metadata('dbg', di_subp)