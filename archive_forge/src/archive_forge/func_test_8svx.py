import warnings
from collections import namedtuple
def test_8svx(h, f):
    """8SVX file"""
    if not h.startswith(b'FORM') or h[8:12] != b'8SVX':
        return None
    return ('8svx', 0, 1, 0, 8)