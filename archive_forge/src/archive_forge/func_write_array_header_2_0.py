import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def write_array_header_2_0(fp, d):
    """ Write the header for an array using the 2.0 format.
        The 2.0 format allows storing very large structured arrays.

    .. versionadded:: 1.9.0

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string
        representation to the header of the file.
    """
    _write_array_header(fp, d, (2, 0))