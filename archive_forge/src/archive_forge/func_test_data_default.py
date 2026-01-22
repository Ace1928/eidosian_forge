from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from .. import load as top_load
from ..imageclasses import spatial_axes_first
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..testing import bytesio_round_trip, deprecated_to, expires, memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
def test_data_default(self):
    img_klass = self.image_class
    hdr_klass = self.image_class.header_class
    data = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    affine = np.eye(4)
    img = img_klass(data, affine)
    self.check_dtypes(data.dtype, img.get_data_dtype())
    header = hdr_klass()
    header.set_data_dtype(np.float32)
    img = img_klass(data, affine, header)
    self.check_dtypes(np.dtype(np.float32), img.get_data_dtype())