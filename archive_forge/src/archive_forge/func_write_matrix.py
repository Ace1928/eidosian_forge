import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
def write_matrix(self, m):
    return _write_data(m, self._fid, self._hb_info)