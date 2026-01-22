from os import PathLike
import warnings
def test_pbm(h, f):
    """PBM (portable bitmap)"""
    if len(h) >= 3 and h[0] == ord(b'P') and (h[1] in b'14') and (h[2] in b' \t\n\r'):
        return 'pbm'