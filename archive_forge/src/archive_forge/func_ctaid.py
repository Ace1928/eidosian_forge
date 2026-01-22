import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def ctaid(self, xyz):
    return call_sreg(self.builder, 'ctaid.%s' % xyz)