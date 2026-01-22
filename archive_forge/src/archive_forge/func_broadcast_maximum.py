from ._internal import NDArrayBase
from ..base import _Null
def broadcast_maximum(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Returns element-wise maximum of the input arrays with broadcasting.

    This function compares two input arrays and returns a new array having the element-wise maxima.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_maximum(x, y) = [[ 1.,  1.,  1.],
                                  [ 1.,  1.,  1.]]



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L80

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