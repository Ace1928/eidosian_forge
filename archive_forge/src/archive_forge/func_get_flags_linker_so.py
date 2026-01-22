import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def get_flags_linker_so(self):
    if os.name == 'nt':
        opt = ['/dll']
    elif self.get_version() >= '9.0':
        opt = ['-shared']
    else:
        opt = ['-K', 'shared']
    return opt