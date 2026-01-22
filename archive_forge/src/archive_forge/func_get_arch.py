from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler
def get_arch(self):
    return ['-xtarget=generic']