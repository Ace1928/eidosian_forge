import math
import struct
from ctypes import create_string_buffer
def tostereo(cp, size, fac1, fac2):
    _check_params(len(cp), size)
    sample_count = _sample_count(cp, size)
    result = create_string_buffer(len(cp) * 2)
    clip = _get_clipfn(size)
    for i in range(sample_count):
        sample = _get_sample(cp, size, i)
        l_sample = clip(sample * fac1)
        r_sample = clip(sample * fac2)
        _put_sample(result, size, i * 2, l_sample)
        _put_sample(result, size, i * 2 + 1, r_sample)
    return result.raw