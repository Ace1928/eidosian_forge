import cupy
from cupy import _core
def unpackbits(a, axis=None, bitorder='big'):
    """Unpacks elements of a uint8 array into a binary-valued output array.

    This function currently does not support ``axis`` option.

    Args:
        a (cupy.ndarray): Input array.
        bitorder (str, optional): bit order to use when unpacking the array,
            allowed values are `'little'` and `'big'`. Defaults to `'big'`.

    Returns:
        cupy.ndarray: The unpacked array.

    .. seealso:: :func:`numpy.unpackbits`
    """
    if a.dtype != cupy.uint8:
        raise TypeError('Expected an input array of unsigned byte data type')
    if axis is not None:
        raise NotImplementedError('axis option is not supported yet')
    if bitorder not in ('big', 'little'):
        raise ValueError("bitorder must be either 'big' or 'little'")
    unpacked = cupy.ndarray(a.size * 8, dtype=cupy.uint8)
    return _unpackbits_kernel[bitorder](a, unpacked)