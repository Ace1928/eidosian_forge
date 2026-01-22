import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
def get_library_dirs(self):
    opt = FCompiler.get_library_dirs(self)
    d = os.environ.get('ABSOFT')
    if d:
        if self.get_version() >= '10.0':
            prefix = 'sh'
        else:
            prefix = ''
        if cpu.is_64bit():
            suffix = '64'
        else:
            suffix = ''
        opt.append(os.path.join(d, '%slib%s' % (prefix, suffix)))
    return opt