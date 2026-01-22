import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
import pytest
from scipy import ndimage
def test_map_coordinates_dts():
    data = np.array([[4, 1, 3, 2], [7, 6, 8, 5], [3, 5, 3, 6]])
    shifted_data = np.array([[0, 0, 0, 0], [0, 4, 1, 3], [0, 7, 6, 8]])
    idx = np.indices(data.shape)
    dts = (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.intp, np.uintp, np.float32, np.float64)
    for order in range(0, 6):
        for data_dt in dts:
            these_data = data.astype(data_dt)
            for coord_dt in dts:
                mat = np.eye(2, dtype=coord_dt)
                off = np.zeros((2,), dtype=coord_dt)
                out = ndimage.affine_transform(these_data, mat, off)
                assert_array_almost_equal(these_data, out)
                coords_m1 = idx.astype(coord_dt) - 1
                coords_p10 = idx.astype(coord_dt) + 10
                out = ndimage.map_coordinates(these_data, coords_m1, order=order)
                assert_array_almost_equal(out, shifted_data)
                out = ndimage.map_coordinates(these_data, coords_p10, order=order)
                assert_array_almost_equal(out, np.zeros((3, 4)))
            out = ndimage.shift(these_data, 1)
            assert_array_almost_equal(out, shifted_data)
            out = ndimage.zoom(these_data, 1)
            assert_array_almost_equal(these_data, out)