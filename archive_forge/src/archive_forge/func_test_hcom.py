import warnings
from collections import namedtuple
def test_hcom(h, f):
    """HCOM file"""
    if h[65:69] != b'FSSD' or h[128:132] != b'HCOM':
        return None
    divisor = get_long_be(h[144:148])
    if divisor:
        rate = 22050 / divisor
    else:
        rate = 0
    return ('hcom', rate, 1, -1, 8)