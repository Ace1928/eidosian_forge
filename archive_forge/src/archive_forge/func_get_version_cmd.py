import os
from numpy.distutils.fcompiler.gnu import GnuFCompiler
def get_version_cmd(self):
    f90 = self.compiler_f90[0]
    d, b = os.path.split(f90)
    vf90 = os.path.join(d, 'v' + b)
    return vf90