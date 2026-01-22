from numpy.testing import assert_, assert_raises
from numpy.lib import NumpyVersion
def test_dev0_version():
    assert_(NumpyVersion('1.9.0.dev0+Unknown') < '1.9.0')
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev0+ffffffff']:
        assert_(NumpyVersion('1.9.0.dev0+f16acvda') < ver)
    assert_(NumpyVersion('1.9.0.dev0+f16acvda') == '1.9.0.dev0+11111111')