from numpy.testing import assert_equal, assert_almost_equal, suppress_warnings
from scipy.misc import face, ascent, electrocardiogram
def test_face():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        assert_equal(face().shape, (768, 1024, 3))