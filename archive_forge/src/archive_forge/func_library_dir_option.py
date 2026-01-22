import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def library_dir_option(self, dir):
    if os.name == 'nt':
        return ['-link', '/PATH:%s' % dir]
    return '-L' + dir