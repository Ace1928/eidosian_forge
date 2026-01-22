import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def update_executables(self):
    f = cyg2win32(dummy_fortran_file())
    self.executables['version_cmd'] = ['<F90>', '-V', '-c', f + '.f', '-o', f + '.o']