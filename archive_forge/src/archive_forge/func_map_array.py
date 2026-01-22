import numpy as np
def map_array(input_arr, input_vals, output_vals, out=None):
    """Map values from input array from input_vals to output_vals.

    Parameters
    ----------
    input_arr : array of int, shape (M[, ...])
        The input label image.
    input_vals : array of int, shape (K,)
        The values to map from.
    output_vals : array, shape (K,)
        The values to map to.
    out: array, same shape as `input_arr`
        The output array. Will be created if not provided. It should
        have the same dtype as `output_vals`.

    Returns
    -------
    out : array, same shape as `input_arr`
        The array of mapped values.

    Notes
    -----
    If `input_arr` contains values that aren't covered by `input_vals`, they
    are set to 0.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski
    >>> ski.util.map_array(
    ...    input_arr=np.array([[0, 2, 2, 0], [3, 4, 5, 0]]),
    ...    input_vals=np.array([1, 2, 3, 4, 6]),
    ...    output_vals=np.array([6, 7, 8, 9, 10]),
    ... )
    array([[0, 7, 7, 0],
           [8, 9, 0, 0]])
    """
    from ._remap import _map_array
    if not np.issubdtype(input_arr.dtype, np.integer):
        raise TypeError('The dtype of an array to be remapped should be integer.')
    orig_shape = input_arr.shape
    input_arr = input_arr.reshape(-1)
    if out is None:
        out = np.empty(orig_shape, dtype=output_vals.dtype)
    elif out.shape != orig_shape:
        raise ValueError(f'If out array is provided, it should have the same shape as the input array. Input array has shape {orig_shape}, provided output array has shape {out.shape}.')
    try:
        out_view = out.view()
        out_view.shape = (-1,)
    except AttributeError:
        raise ValueError(f'If out array is provided, it should be either contiguous or 1-dimensional. Got array with shape {out.shape} and strides {out.strides}.')
    input_vals = input_vals.astype(input_arr.dtype, copy=False)
    output_vals = output_vals.astype(out.dtype, copy=False)
    _map_array(input_arr, out_view, input_vals, output_vals)
    return out