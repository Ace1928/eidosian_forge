import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def mmwrite(target, a, comment=None, field=None, precision=None, symmetry='AUTO'):
    """
    Writes the sparse or dense array `a` to Matrix Market file-like `target`.

    Parameters
    ----------
    target : str or file-like
        Matrix Market filename (extension .mtx) or open file-like object.
    a : array like
        Sparse or dense 2-D array.
    comment : str, optional
        Comments to be prepended to the Matrix Market file.
    field : None or str, optional
        Either 'real', 'complex', 'pattern', or 'integer'.
    precision : None or int, optional
        Number of digits to display for real or complex values.
    symmetry : None or str, optional
        Either 'AUTO', 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
        If symmetry is None the symmetry type of 'a' is determined by its
        values. If symmetry is 'AUTO' the symmetry type of 'a' is either
        determined or set to 'general', at mmwrite's discretion.

    Returns
    -------
    None

    Notes
    -----
    .. versionchanged:: 1.12.0
        C++ implementation.

    Examples
    --------
    >>> from io import BytesIO
    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix
    >>> from scipy.io import mmwrite

    Write a small NumPy array to a matrix market file.  The file will be
    written in the ``'array'`` format.

    >>> a = np.array([[1.0, 0, 0, 0], [0, 2.5, 0, 6.25]])
    >>> target = BytesIO()
    >>> mmwrite(target, a)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    2 4
    1
    0
    0
    2.5
    0
    0
    0
    6.25

    Add a comment to the output file, and set the precision to 3.

    >>> target = BytesIO()
    >>> mmwrite(target, a, comment='\\n Some test data.\\n', precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    % Some test data.
    %
    2 4
    1.00e+00
    0.00e+00
    0.00e+00
    2.50e+00
    0.00e+00
    0.00e+00
    0.00e+00
    6.25e+00

    Convert to a sparse matrix before calling ``mmwrite``.  This will
    result in the output format being ``'coordinate'`` rather than
    ``'array'``.

    >>> target = BytesIO()
    >>> mmwrite(target, coo_matrix(a), precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix coordinate real general
    %
    2 4 3
    1 1 1.00e+00
    2 2 2.50e+00
    2 4 6.25e+00

    Write a complex Hermitian array to a matrix market file.  Note that
    only six values are actually written to the file; the other values
    are implied by the symmetry.

    >>> z = np.array([[3, 1+2j, 4-3j], [1-2j, 1, -5j], [4+3j, 5j, 2.5]])
    >>> z
    array([[ 3. +0.j,  1. +2.j,  4. -3.j],
           [ 1. -2.j,  1. +0.j, -0. -5.j],
           [ 4. +3.j,  0. +5.j,  2.5+0.j]])

    >>> target = BytesIO()
    >>> mmwrite(target, z, precision=2)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array complex hermitian
    %
    3 3
    3.0e+00 0.0e+00
    1.0e+00 -2.0e+00
    4.0e+00 3.0e+00
    1.0e+00 0.0e+00
    0.0e+00 5.0e+00
    2.5e+00 0.0e+00

    This method is threaded.
    The default number of threads is equal to the number of CPUs in the system.
    Use `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ to override:

    >>> import threadpoolctl
    >>>
    >>> target = BytesIO()
    >>> with threadpoolctl.threadpool_limits(limits=2):
    ...     mmwrite(target, a)

    """
    from . import _fmm_core
    if isinstance(a, list) or isinstance(a, tuple) or hasattr(a, '__array__'):
        a = np.asarray(a)
    if symmetry == 'AUTO':
        if ALWAYS_FIND_SYMMETRY or (hasattr(a, 'shape') and max(a.shape) < 100):
            symmetry = None
        else:
            symmetry = 'general'
    if symmetry is None:
        symmetry = _mmio.MMFile()._get_symmetry(a)
    symmetry = _validate_symmetry(symmetry)
    cursor = _get_write_cursor(target, comment=comment, precision=precision, symmetry=symmetry)
    if isinstance(a, np.ndarray):
        a = _apply_field(a, field, no_pattern=True)
        _fmm_core.write_body_array(cursor, a)
    elif scipy.sparse.issparse(a):
        a = a.tocoo()
        if symmetry is not None and symmetry != 'general':
            from scipy.sparse import coo_array
            lower_triangle_mask = a.row >= a.col
            a = coo_array((a.data[lower_triangle_mask], (a.row[lower_triangle_mask], a.col[lower_triangle_mask])), shape=a.shape)
        data = _apply_field(a.data, field)
        _fmm_core.write_body_coo(cursor, a.shape, a.row, a.col, data)
    else:
        raise ValueError('unknown matrix type: %s' % type(a))