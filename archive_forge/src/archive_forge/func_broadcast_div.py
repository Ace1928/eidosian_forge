from ._internal import NDArrayBase
from ..base import _Null
def broadcast_div(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Returns element-wise division of the input arrays with broadcasting.

    Example::

       x = [[ 6.,  6.,  6.],
            [ 6.,  6.,  6.]]

       y = [[ 2.],
            [ 3.]]

       broadcast_div(x, y) = [[ 3.,  3.,  3.],
                              [ 2.,  2.,  2.]]

    Supported sparse operations:

       broadcast_div(csr, dense(1D)) = csr



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L186

    Parameters
    ----------
    lhs : NDArray
        First input to the function
    rhs : NDArray
        Second input to the function

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)