import cupy
import operator
import warnings
def trimseq(seq):
    """Removes small polynomial series coefficients.

    Args:
        seq (cupy.ndarray): input array.

    Returns:
        cupy.ndarray: input array with trailing zeros removed. If the
        resulting output is empty, it returns the first element.

    .. seealso:: :func:`numpy.polynomial.polyutils.trimseq`

    """
    if seq.size == 0:
        return seq
    ret = cupy.trim_zeros(seq, trim='b')
    if ret.size > 0:
        return ret
    return seq[:1]