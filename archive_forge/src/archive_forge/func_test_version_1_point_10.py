from numpy.testing import assert_, assert_raises
from numpy.lib import NumpyVersion
def test_version_1_point_10():
    assert_(NumpyVersion('1.9.0') < '1.10.0')
    assert_(NumpyVersion('1.11.0') < '1.11.1')
    assert_(NumpyVersion('1.11.0') == '1.11.0')
    assert_(NumpyVersion('1.99.11') < '1.99.12')