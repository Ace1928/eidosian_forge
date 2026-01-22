from the public API.  This format is called packed.  When packed,
import io
import itertools
import math
import operator
import struct
import sys
import zlib
import warnings
from array import array
from functools import reduce
from pygame.tests.test_utils import tostring
fromarray = from_array
import tempfile
import unittest
def convert_ppm_and_pgm(self, ppmfile, pgmfile, outfile):
    """
        Convert a PPM and PGM file containing raw pixel data into a
        PNG outfile with the parameters set in the writer object.
        """
    pixels = array('B')
    pixels.fromfile(ppmfile, self.bitdepth / 8 * self.color_planes * self.width * self.height)
    apixels = array('B')
    apixels.fromfile(pgmfile, self.bitdepth / 8 * self.width * self.height)
    pixels = interleave_planes(pixels, apixels, self.bitdepth / 8 * self.color_planes, self.bitdepth / 8)
    if self.interlace:
        self.write_passes(outfile, self.array_scanlines_interlace(pixels))
    else:
        self.write_passes(outfile, self.array_scanlines(pixels))