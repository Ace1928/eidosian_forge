import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def mmread(source):
    """
    Reads the contents of a Matrix Market file-like 'source' into a matrix.

    Parameters
    ----------
    source : str or file-like
        Matrix Market filename (extensions .mtx, .mtz.gz)
        or open file-like object.

    Returns
    -------
    a : ndarray or coo_matrix
        Dense or sparse matrix depending on the matrix format in the
        Matrix Market file.

    Notes
    -----
    .. versionchanged:: 1.12.0
        C++ implementation.

    Examples
    --------
    >>> from io import StringIO
    >>> from scipy.io import mmread

    >>> text = '''%%MatrixMarket matrix coordinate real general
    ...  5 5 7
    ...  2 3 1.0
    ...  3 4 2.0
    ...  3 5 3.0
    ...  4 1 4.0
    ...  4 2 5.0
    ...  4 3 6.0
    ...  4 4 7.0
    ... '''

    ``mmread(source)`` returns the data as sparse matrix in COO format.

    >>> m = mmread(StringIO(text))
    >>> m
    <5x5 sparse matrix of type '<class 'numpy.float64'>'
    with 7 stored elements in COOrdinate format>
    >>> m.A
    array([[0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 2., 3.],
           [4., 5., 6., 7., 0.],
           [0., 0., 0., 0., 0.]])

    This method is threaded.
    The default number of threads is equal to the number of CPUs in the system.
    Use `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ to override:

    >>> import threadpoolctl
    >>>
    >>> with threadpoolctl.threadpool_limits(limits=2):
    ...     m = mmread(StringIO(text))

    """
    cursor, stream_to_close = _get_read_cursor(source)
    if cursor.header.format == 'array':
        mat = _read_body_array(cursor)
        if stream_to_close:
            stream_to_close.close()
        return mat
    else:
        from scipy.sparse import coo_matrix
        triplet, shape = _read_body_coo(cursor, generalize_symmetry=True)
        if stream_to_close:
            stream_to_close.close()
        return coo_matrix(triplet, shape=shape)