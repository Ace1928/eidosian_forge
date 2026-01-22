from __future__ import annotations
from . import Image
def lighter(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Compares the two images, pixel by pixel, and returns a new image containing
    the lighter values. ::

        out = max(image1, image2)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    image1.load()
    image2.load()
    return image1._new(image1.im.chop_lighter(image2.im))