from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import MGHImage, Nifti1Image, Nifti1Pair, all_image_classes
from ..fileholders import FileHolderError
from ..spatialimages import SpatialImage
def test_files_spatialimages():
    arr = np.zeros((2, 3, 4))
    aff = np.eye(4)
    klasses = [klass for klass in all_image_classes if klass.rw and issubclass(klass, SpatialImage)]
    for klass in klasses:
        file_map = klass.make_file_map()
        for key, value in file_map.items():
            assert value.filename is None
            assert value.fileobj is None
            assert value.pos == 0
        if not klass.makeable:
            continue
        if klass == MGHImage:
            img = klass(arr.astype(np.float32), aff)
        else:
            img = klass(arr, aff)
        for key, value in img.file_map.items():
            assert value.filename is None
            assert value.fileobj is None
            assert value.pos == 0