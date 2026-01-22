import warnings
from collections import namedtuple
def test_sndt(h, f):
    """SNDT file"""
    if h.startswith(b'SOUND'):
        nsamples = get_long_le(h[8:12])
        rate = get_short_le(h[20:22])
        return ('sndt', rate, 1, nsamples, 8)