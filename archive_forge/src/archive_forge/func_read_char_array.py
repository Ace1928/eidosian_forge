import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
def read_char_array(self, hdr):
    """ latin-1 text matrix (char matrix) reader

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ndarray
            with dtype 'U1', shape given by `hdr` ``dims``
        """
    arr = self.read_sub_array(hdr).astype(np.uint8)
    S = arr.tobytes().decode('latin-1')
    return np.ndarray(shape=hdr.dims, dtype=np.dtype('U1'), buffer=np.array(S)).copy()