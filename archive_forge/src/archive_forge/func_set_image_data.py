import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def set_image_data(self, array):
    assert isinstance(array, numpy.ndarray)
    shape = array.shape
    dtype = array.dtype
    with self._fi as lib:
        isle = lib.FreeImage_IsLittleEndian()
    r, c = shape[:2]
    if len(shape) == 2:
        n_channels = 1
        w_shape = (c, r)
    elif len(shape) == 3:
        n_channels = shape[2]
        w_shape = (n_channels, c, r)
    else:
        n_channels = shape[0]

    def n(arr):
        return arr[::-1].T
    wrapped_array = self._wrap_bitmap_bits_in_array(w_shape, dtype, True)
    if len(shape) == 3 and isle and (dtype.type == numpy.uint8):
        R = array[:, :, 0]
        G = array[:, :, 1]
        B = array[:, :, 2]
        wrapped_array[0] = n(B)
        wrapped_array[1] = n(G)
        wrapped_array[2] = n(R)
        if shape[2] == 4:
            A = array[:, :, 3]
            wrapped_array[3] = n(A)
    else:
        wrapped_array[:] = n(array)
    if self._need_finish:
        self._finish_wrapped_array(wrapped_array)
    if len(shape) == 2 and dtype.type == numpy.uint8:
        with self._fi as lib:
            palette = lib.FreeImage_GetPalette(self._bitmap)
        palette = ctypes.c_void_p(palette)
        if not palette:
            raise RuntimeError('Could not get image palette')
        try:
            palette_data = GREY_PALETTE.ctypes.data
        except Exception:
            palette_data = GREY_PALETTE.__array_interface__['data'][0]
        ctypes.memmove(palette, palette_data, 1024)