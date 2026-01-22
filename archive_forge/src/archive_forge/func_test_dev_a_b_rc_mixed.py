from numpy.testing import assert_, assert_raises
from numpy.lib import NumpyVersion
def test_dev_a_b_rc_mixed():
    assert_(NumpyVersion('1.9.0a2.dev-f16acvda') == '1.9.0a2.dev-11111111')
    assert_(NumpyVersion('1.9.0a2.dev-6acvda54') < '1.9.0a2')