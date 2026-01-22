import warnings
from collections import namedtuple
def test_voc(h, f):
    """VOC file"""
    if not h.startswith(b'Creative Voice File\x1a'):
        return None
    sbseek = get_short_le(h[20:22])
    rate = 0
    if 0 <= sbseek < 500 and h[sbseek] == 1:
        ratecode = 256 - h[sbseek + 4]
        if ratecode:
            rate = int(1000000.0 / ratecode)
    return ('voc', rate, 1, -1, 8)