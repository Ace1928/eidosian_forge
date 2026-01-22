import numba
import numpy as np
def rescale_and_lookup1d(data, scale, offset, lut):
    data_out = np.empty_like(data, dtype=lut.dtype)
    _rescale_and_lookup1d_function(data, float(scale), float(offset), lut, data_out)
    return data_out