from ._internal import NDArrayBase
from ..base import _Null
def index_array(data=None, axes=_Null, out=None, name=None, **kwargs):
    """Returns an array of indexes of the input array.

    For an input array with shape  :math:`(d_1, d_2, ..., d_n)`, `index_array` returns a
    :math:`(d_1, d_2, ..., d_n, n)` array `idx`, where
    :math:`idx[i_1, i_2, ..., i_n, :] = [i_1, i_2, ..., i_n]`.

    Additionally, when the parameter `axes` is specified, `idx` will be a
    :math:`(d_1, d_2, ..., d_n, m)` array where `m` is the length of `axes`, and the following
    equality will hold: :math:`idx[i_1, i_2, ..., i_n, j] = i_{axes[j]}`.

    Examples::

        x = mx.nd.ones((3, 2))

        mx.nd.contrib.index_array(x) = [[[0 0]
                                         [0 1]]

                                        [[1 0]
                                         [1 1]]

                                        [[2 0]
                                         [2 1]]]

        x = mx.nd.ones((3, 2, 2))

        mx.nd.contrib.index_array(x, axes=(1, 0)) = [[[[0 0]
                                                       [0 0]]

                                                      [[1 0]
                                                       [1 0]]]


                                                     [[[0 1]
                                                       [0 1]]

                                                      [[1 1]
                                                       [1 1]]]


                                                     [[[0 2]
                                                       [0 2]]

                                                      [[1 2]
                                                       [1 2]]]]



    Defined in ../src/operator/contrib/index_array.cc:L118

    Parameters
    ----------
    data : NDArray
        Input data
    axes : Shape or None, optional, default=None
        The axes to include in the index array. Supports negative values.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)