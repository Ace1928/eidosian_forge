import cupy
from cupy import _core
def packbits(a, axis=None, bitorder='big'):
    """Packs the elements of a binary-valued array into bits in a uint8 array.

    This function currently does not support ``axis`` option.

    Args:
        a (cupy.ndarray): Input array.
        axis (int, optional): Not supported yet.
        bitorder (str, optional): bit order to use when packing the array,
            allowed values are `'little'` and `'big'`. Defaults to `'big'`.

    Returns:
        cupy.ndarray: The packed array.

    .. note::
        When the input array is empty, this function returns a copy of it,
        i.e., the type of the output array is not necessarily always uint8.
        This exactly follows the NumPy's behaviour (as of version 1.11),
        alghough this is inconsistent to the documentation.

    .. seealso:: :func:`numpy.packbits`
    """
    if a.dtype.kind not in 'biu':
        raise TypeError('Expected an input array of integer or boolean data type')
    if axis is not None:
        raise NotImplementedError('axis option is not supported yet')
    if bitorder not in ('big', 'little'):
        raise ValueError("bitorder must be either 'big' or 'little'")
    a = a.ravel()
    packed_size = (a.size + 7) // 8
    packed = cupy.zeros((packed_size,), dtype=cupy.uint8)
    return _packbits_kernel[bitorder](a, a.size, packed)