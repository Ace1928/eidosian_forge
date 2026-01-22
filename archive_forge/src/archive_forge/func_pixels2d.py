from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def pixels2d(surface):
    """pygame.surfarray.pixels2d(Surface): return array

    reference pixels into a 2d array

    Create a new 2D array that directly references the pixel values in a
    Surface. Any changes to the array will affect the pixels in the
    Surface. This is a fast operation since no data is copied.

    Pixels from a 24-bit Surface cannot be referenced, but all other
    Surface bit depths can.

    The Surface this references will remain locked for the lifetime of
    the array (see the Surface.lock - lock the Surface memory for pixel
    access method).
    """
    if surface.get_bitsize() not in _pixel2d_bitdepths:
        raise ValueError('unsupported bit depth for 2D reference array')
    try:
        return numpy_array(surface.get_view('2'), copy=False)
    except (ValueError, TypeError):
        raise ValueError(f'bit depth {surface.get_bitsize()} unsupported for 2D reference array')