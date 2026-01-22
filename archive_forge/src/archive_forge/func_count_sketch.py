from ._internal import NDArrayBase
from ..base import _Null
def count_sketch(data=None, h=None, s=None, out_dim=_Null, processing_batch_size=_Null, out=None, name=None, **kwargs):
    """Apply CountSketch to input: map a d-dimension data to k-dimension data"

    .. note:: `count_sketch` is only available on GPU.

    Assume input data has shape (N, d), sign hash table s has shape (N, d),
    index hash table h has shape (N, d) and mapping dimension out_dim = k,
    each element in s is either +1 or -1, each element in h is random integer from 0 to k-1.
    Then the operator computs:

    .. math::
       out[h[i]] += data[i] * s[i]

    Example::

       out_dim = 5
       x = [[1.2, 2.5, 3.4],[3.2, 5.7, 6.6]]
       h = [[0, 3, 4]]
       s = [[1, -1, 1]]
       mx.contrib.ndarray.count_sketch(data=x, h=h, s=s, out_dim = 5) = [[1.2, 0, 0, -2.5, 3.4],
                                                                         [3.2, 0, 0, -5.7, 6.6]]



    Defined in ../src/operator/contrib/count_sketch.cc:L66

    Parameters
    ----------
    data : NDArray
        Input data to the CountSketchOp.
    h : NDArray
        The index vector
    s : NDArray
        The sign vector
    out_dim : int, required
        The output dimension.
    processing_batch_size : int, optional, default='32'
        How many sketch vectors to process at one time.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)