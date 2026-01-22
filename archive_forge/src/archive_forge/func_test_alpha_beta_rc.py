from numpy.testing import assert_, assert_raises
from numpy.lib import NumpyVersion
def test_alpha_beta_rc():
    assert_(NumpyVersion('1.8.0rc1') == '1.8.0rc1')
    for ver in ['1.8.0', '1.8.0rc2']:
        assert_(NumpyVersion('1.8.0rc1') < ver)
    for ver in ['1.8.0a2', '1.8.0b3', '1.7.2rc4']:
        assert_(NumpyVersion('1.8.0rc1') > ver)
    assert_(NumpyVersion('1.8.0b1') > '1.8.0a2')