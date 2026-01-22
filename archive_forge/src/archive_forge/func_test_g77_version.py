from numpy.testing import assert_
import numpy.distutils.fcompiler
def test_g77_version(self):
    fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu')
    for vs, version in g77_version_strings:
        v = fc.version_match(vs)
        assert_(v == version, (vs, v))