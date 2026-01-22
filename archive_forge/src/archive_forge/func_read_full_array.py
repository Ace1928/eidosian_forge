import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
def read_full_array(self, hdr):
    """ Full (rather than sparse) matrix getter

        Read matrix (array) can be real or complex

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ndarray
            complex array if ``hdr.is_complex`` is True, otherwise a real
            numeric array
        """
    if hdr.is_complex:
        res = self.read_sub_array(hdr, copy=False)
        res_j = self.read_sub_array(hdr, copy=False)
        return res + res_j * 1j
    return self.read_sub_array(hdr)