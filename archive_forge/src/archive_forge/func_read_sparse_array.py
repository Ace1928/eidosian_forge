import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
def read_sparse_array(self, hdr):
    """ Read and return sparse matrix type

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ``scipy.sparse.coo_matrix``
            with dtype ``float`` and shape read from the sparse matrix data

        Notes
        -----
        MATLAB 4 real sparse arrays are saved in a N+1 by 3 array format, where
        N is the number of non-zero values. Column 1 values [0:N] are the
        (1-based) row indices of the each non-zero value, column 2 [0:N] are the
        column indices, column 3 [0:N] are the (real) values. The last values
        [-1,0:2] of the rows, column indices are shape[0] and shape[1]
        respectively of the output matrix. The last value for the values column
        is a padding 0. mrows and ncols values from the header give the shape of
        the stored matrix, here [N+1, 3]. Complex data are saved as a 4 column
        matrix, where the fourth column contains the imaginary component; the
        last value is again 0. Complex sparse data do *not* have the header
        ``imagf`` field set to True; the fact that the data are complex is only
        detectable because there are 4 storage columns.
        """
    res = self.read_sub_array(hdr)
    tmp = res[:-1, :]
    dims = (int(res[-1, 0]), int(res[-1, 1]))
    I = np.ascontiguousarray(tmp[:, 0], dtype='intc')
    J = np.ascontiguousarray(tmp[:, 1], dtype='intc')
    I -= 1
    J -= 1
    if res.shape[1] == 3:
        V = np.ascontiguousarray(tmp[:, 2], dtype='float')
    else:
        V = np.ascontiguousarray(tmp[:, 2], dtype='complex')
        V.imag = tmp[:, 3]
    return scipy.sparse.coo_matrix((V, (I, J)), dims)