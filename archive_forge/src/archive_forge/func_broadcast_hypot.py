from ._internal import NDArrayBase
from ..base import _Null
def broadcast_hypot(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """ Returns the hypotenuse of a right angled triangle, given its "legs"
    with broadcasting.

    It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.

    Example::

       x = [[ 3.,  3.,  3.]]

       y = [[ 4.],
            [ 4.]]

       broadcast_hypot(x, y) = [[ 5.,  5.,  5.],
                                [ 5.,  5.,  5.]]

       z = [[ 0.],
            [ 4.]]

       broadcast_hypot(x, z) = [[ 3.,  3.,  3.],
                                [ 5.,  5.,  5.]]



    Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L157

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