import math
import struct
from ctypes import create_string_buffer
def tomono(cp, size, fac1, fac2):
    _check_params(len(cp), size)
    clip = _get_clipfn(size)
    sample_count = _sample_count(cp, size)
    result = create_string_buffer(len(cp) / 2)
    for i in range(0, sample_count, 2):
        l_sample = getsample(cp, size, i)
        r_sample = getsample(cp, size, i + 1)
        sample = l_sample * fac1 + r_sample * fac2
        sample = clip(sample)
        _put_sample(result, size, i / 2, sample)
    return result.raw