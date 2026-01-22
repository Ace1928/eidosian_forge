from numpy.testing import assert_
import numpy.distutils.fcompiler
def test_not_g77(self):
    fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu')
    for vs, _ in gfortran_version_strings:
        v = fc.version_match(vs)
        assert_(v is None, (vs, v))