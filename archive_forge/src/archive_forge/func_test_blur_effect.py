from numpy.testing import assert_array_equal
from skimage.color import rgb2gray
from skimage.data import astronaut, cells3d
from skimage.filters import gaussian
from skimage.measure import blur_effect
def test_blur_effect():
    """Test that the blur metric increases with more blurring."""
    image = astronaut()
    B0 = blur_effect(image, channel_axis=-1)
    B1 = blur_effect(gaussian(image, sigma=1, channel_axis=-1), channel_axis=-1)
    B2 = blur_effect(gaussian(image, sigma=4, channel_axis=-1), channel_axis=-1)
    assert 0 <= B0 < 1
    assert B0 < B1 < B2