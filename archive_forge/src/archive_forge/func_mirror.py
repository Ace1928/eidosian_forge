from __future__ import annotations
import functools
import operator
import re
from . import ExifTags, Image, ImagePalette
def mirror(image):
    """
    Flip image horizontally (left to right).

    :param image: The image to mirror.
    :return: An image.
    """
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)