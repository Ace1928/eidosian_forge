from numpy.testing import assert_equal, assert_almost_equal, suppress_warnings
from scipy.misc import face, ascent, electrocardiogram
def test_electrocardiogram():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        ecg = electrocardiogram()
        assert ecg.dtype == float
        assert_equal(ecg.shape, (108000,))
        assert_almost_equal(ecg.mean(), -0.16510875)
        assert_almost_equal(ecg.std(), 0.5992473991177294)