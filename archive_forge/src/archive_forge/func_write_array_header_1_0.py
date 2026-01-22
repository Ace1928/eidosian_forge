import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def write_array_header_1_0(fp, d):
    """ Write the header for an array using the 1.0 format.

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string
        representation to the header of the file.
    """
    _write_array_header(fp, d, (1, 0))