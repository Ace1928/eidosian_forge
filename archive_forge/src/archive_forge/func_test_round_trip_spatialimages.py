from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import MGHImage, Nifti1Image, Nifti1Pair, all_image_classes
from ..fileholders import FileHolderError
from ..spatialimages import SpatialImage
def test_round_trip_spatialimages():
    data = np.arange(24, dtype='i4').reshape((2, 3, 4))
    aff = np.eye(4)
    klasses = [klass for klass in all_image_classes if klass.rw and klass.makeable and issubclass(klass, SpatialImage)]
    for klass in klasses:
        file_map = klass.make_file_map()
        for key in file_map:
            file_map[key].fileobj = BytesIO()
        img = klass(data, aff)
        img.file_map = file_map
        img.to_file_map()
        img2 = klass.from_file_map(file_map)
        assert_array_equal(img2.get_fdata(), data)
        img2.to_file_map()
        img3 = klass.from_file_map(file_map)
        assert_array_equal(img3.get_fdata(), data)