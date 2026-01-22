import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
def hb_read(path_or_open_file):
    """Read HB-format file.

    Parameters
    ----------
    path_or_open_file : path-like or file-like
        If a file-like object, it is used as-is. Otherwise, it is opened
        before reading.

    Returns
    -------
    data : scipy.sparse.csc_matrix instance
        The data read from the HB file as a sparse matrix.

    Notes
    -----
    At the moment not the full Harwell-Boeing format is supported. Supported
    features are:

        - assembled, non-symmetric, real matrices
        - integer for pointer/indices
        - exponential format for float values, and int format

    Examples
    --------
    We can read and write a harwell-boeing format file:

    >>> from scipy.io import hb_read, hb_write
    >>> from scipy.sparse import csr_matrix, eye
    >>> data = csr_matrix(eye(3))  # create a sparse matrix
    >>> hb_write("data.hb", data)  # write a hb file
    >>> print(hb_read("data.hb"))  # read a hb file
      (0, 0)	1.0
      (1, 1)	1.0
      (2, 2)	1.0

    """

    def _get_matrix(fid):
        hb = HBFile(fid)
        return hb.read_matrix()
    if hasattr(path_or_open_file, 'read'):
        return _get_matrix(path_or_open_file)
    else:
        with open(path_or_open_file) as f:
            return _get_matrix(f)