import re
from . import utilities
def get_manifold_from_file(filename):
    """
    As get_manifold but takes filename. Returns a byte sequence.
    """
    return get_manifold(open(filename, 'rb').read())