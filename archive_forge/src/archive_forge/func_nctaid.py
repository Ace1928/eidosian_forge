import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def nctaid(self, xyz):
    return call_sreg(self.builder, 'nctaid.%s' % xyz)