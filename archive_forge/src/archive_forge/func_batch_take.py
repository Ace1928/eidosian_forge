from ._internal import NDArrayBase
from ..base import _Null
def batch_take(a=None, indices=None, out=None, name=None, **kwargs):
    """Takes elements from a data batch.

    .. note::
      `batch_take` is deprecated. Use `pick` instead.

    Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
    an output array of shape ``(i0,)`` with::

      output[i] = input[i, indices[i]]

    Examples::

      x = [[ 1.,  2.],
           [ 3.,  4.],
           [ 5.,  6.]]

      // takes elements with specified indices
      batch_take(x, [0,1,0]) = [ 1.  4.  5.]



    Defined in ../src/operator/tensor/indexing_op.cc:L835

    Parameters
    ----------
    a : NDArray
        The input array
    indices : NDArray
        The index array

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)