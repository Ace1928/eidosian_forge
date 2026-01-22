from ._internal import NDArrayBase
from ..base import _Null
Return an array of zeros with the same shape, type and storage type
    as the input array.

    The storage type of ``zeros_like`` output depends on the storage type of the input

    - zeros_like(row_sparse) = row_sparse
    - zeros_like(csr) = csr
    - zeros_like(default) = default

    Examples::

      x = [[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]]

      zeros_like(x) = [[ 0.,  0.,  0.],
                       [ 0.,  0.,  0.]]



    Parameters
    ----------
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    