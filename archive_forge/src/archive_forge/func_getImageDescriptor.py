import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
def getImageDescriptor(self, im, xy=None):
    """Used for the local color table properties per image.
        Otherwise global color table applies to all frames irrespective of
        whether additional colors comes in play that require a redefined
        palette. Still a maximum of 256 color per frame, obviously.

        Written by Ant1 on 2010-08-22
        Modified by Alex Robinson in Janurari 2011 to implement subrectangles.
        """
    if xy is None:
        xy = (0, 0)
    bb = b','
    bb += intToBin(xy[0])
    bb += intToBin(xy[1])
    bb += intToBin(im.size[0])
    bb += intToBin(im.size[1])
    bb += b'\x87'
    return bb