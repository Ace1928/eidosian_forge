import numpy.distutils.fcompiler
from numpy.testing import assert_
def test_64bit_version(self):
    fc = numpy.distutils.fcompiler.new_fcompiler(compiler='intelem')
    for vs, version in intel_64bit_version_strings:
        v = fc.version_match(vs)
        assert_(v == version)