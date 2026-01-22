from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from .. import load as top_load
from ..imageclasses import spatial_axes_first
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..testing import bytesio_round_trip, deprecated_to, expires, memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
def test_isolation(self):
    img_klass = self.image_class
    arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    aff = np.eye(4)
    img = img_klass(arr, aff)
    assert (img.affine == aff).all()
    aff[0, 0] = 99
    assert not np.all(img.affine == aff)
    ihdr = img.header
    img = img_klass(arr, aff, ihdr)
    ihdr.set_zooms((4, 5, 6))
    assert img.header != ihdr