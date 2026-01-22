import numpy as np
from .structreader import Unpacker
from .utils import find_private_section
def nt_str(s):
    """Strip string to first null

    Parameters
    ----------
    s : bytes

    Returns
    -------
    sdash : str
       s stripped to first occurrence of null (0)
    """
    zero_pos = s.find(b'\x00')
    if zero_pos == -1:
        return s
    return s[:zero_pos].decode('latin-1')