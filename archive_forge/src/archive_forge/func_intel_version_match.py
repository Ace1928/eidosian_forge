import sys
from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
def intel_version_match(type):
    return simple_version_match(start='Intel.*?Fortran.*?(?:%s).*?Version' % (type,))