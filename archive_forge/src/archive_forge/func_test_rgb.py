from os import PathLike
import warnings
def test_rgb(h, f):
    """SGI image library"""
    if h.startswith(b'\x01\xda'):
        return 'rgb'