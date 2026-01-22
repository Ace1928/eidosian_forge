from ._internal import NDArrayBase
from ..base import _Null
def elemwise_add(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Adds arguments element-wise.

    The storage type of ``elemwise_add`` output depends on storage types of inputs

       - elemwise_add(row_sparse, row_sparse) = row_sparse
       - elemwise_add(csr, csr) = csr
       - elemwise_add(default, csr) = default
       - elemwise_add(csr, default) = default
       - elemwise_add(default, rsp) = default
       - elemwise_add(rsp, default) = default
       - otherwise, ``elemwise_add`` generates output with default storage



    Parameters
    ----------
    lhs : NDArray
        first input
    rhs : NDArray
        second input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)