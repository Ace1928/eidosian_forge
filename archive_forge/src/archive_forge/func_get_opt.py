from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler
def get_opt(self):
    return ['-fast', '-dalign']