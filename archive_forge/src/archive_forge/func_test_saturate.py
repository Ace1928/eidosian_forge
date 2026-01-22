from __future__ import annotations
import numpy as np
from datashader.composite import add, saturate, over, source
def test_saturate():
    o = src.copy()
    o[0, 1] = 0
    np.testing.assert_equal(saturate(src, clear), o)
    np.testing.assert_equal(saturate(src, clear_white), o)
    o = np.full((3, 3), white, dtype='uint32')
    np.testing.assert_equal(saturate(src, white), o)
    o = np.full((3, 3), blue, dtype='uint32')
    np.testing.assert_equal(saturate(src, blue), o)
    o = np.array([[2113863680, 2113863680, 4294935170], [4211015680, 4202659584, 4202627199], [4294901760, 4286382080, 3082818323]])
    np.testing.assert_equal(saturate(src, half_blue), o)
    o = np.array([[2105344125, 2105344125, 4290740927], [4206755902, 4198399806, 4198367422], [4290707517, 4282187837, 3077051240]])
    np.testing.assert_equal(saturate(src, half_purple), o)