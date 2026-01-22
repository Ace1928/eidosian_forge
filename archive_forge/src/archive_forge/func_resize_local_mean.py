import numpy as np
from scipy import ndimage as ndi
from ._geometric import SimilarityTransform, AffineTransform, ProjectiveTransform
from ._warps_cy import _warp_fast
from ..measure import block_reduce
from .._shared.utils import (
def resize_local_mean(image, output_shape, grid_mode=True, preserve_range=False, *, channel_axis=None):
    """Resize an array with the local mean / bilinear scaling.

    Parameters
    ----------
    image : ndarray
        Input image. If this is a multichannel image, the axis corresponding
        to channels should be specified using `channel_axis`.
    output_shape : iterable
        Size of the generated output image. When `channel_axis` is not None,
        the `channel_axis` should either be omitted from `output_shape` or the
        ``output_shape[channel_axis]`` must match
        ``image.shape[channel_axis]``. If the length of `output_shape` exceeds
        image.ndim, additional singleton dimensions will be appended to the
        input ``image`` as needed.
    grid_mode : bool, optional
        Defines ``image`` pixels position: if True, pixels are assumed to be at
        grid intersections, otherwise at cell centers. As a consequence,
        for example, a 1d signal of length 5 is considered to have length 4
        when `grid_mode` is False, but length 5 when `grid_mode` is True. See
        the following visual illustration:

        .. code-block:: text

                | pixel 1 | pixel 2 | pixel 3 | pixel 4 | pixel 5 |
                     |<-------------------------------------->|
                                        vs.
                |<----------------------------------------------->|

        The starting point of the arrow in the diagram above corresponds to
        coordinate location 0 in each mode.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    resized : ndarray
        Resized version of the input.

    See Also
    --------
    resize, downscale_local_mean

    Notes
    -----
    This method is sometimes referred to as "area-based" interpolation or
    "pixel mixing" interpolation [1]_. When `grid_mode` is True, it is
    equivalent to using OpenCV's resize with `INTER_AREA` interpolation mode.
    It is commonly used for image downsizing. If the downsizing factors are
    integers, then `downscale_local_mean` should be preferred instead.

    References
    ----------
    .. [1] http://entropymine.com/imageworsener/pixelmixing/

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.transform import resize_local_mean
    >>> image = data.camera()
    >>> resize_local_mean(image, (100, 100)).shape
    (100, 100)

    """
    if channel_axis is not None:
        if channel_axis < -image.ndim or channel_axis >= image.ndim:
            raise ValueError('invalid channel_axis')
        image = np.moveaxis(image, channel_axis, -1)
        nc = image.shape[-1]
        output_ndim = len(output_shape)
        if output_ndim == image.ndim - 1:
            output_shape = output_shape + (nc,)
        elif output_ndim == image.ndim:
            if output_shape[channel_axis] != nc:
                raise ValueError('Cannot reshape along the channel_axis. Use channel_axis=None to reshape along all axes.')
            channel_axis = channel_axis % image.ndim
            output_shape = output_shape[:channel_axis] + output_shape[channel_axis:] + (nc,)
        else:
            raise ValueError('len(output_shape) must be image.ndim or (image.ndim - 1) when a channel_axis is specified.')
        resized = image
    else:
        resized, output_shape = _preprocess_resize_output_shape(image, output_shape)
    resized = convert_to_float(resized, preserve_range)
    dtype = resized.dtype
    for axis, (old_size, new_size) in enumerate(zip(image.shape, output_shape)):
        if old_size == new_size:
            continue
        weights = _local_mean_weights(old_size, new_size, grid_mode, dtype)
        product = np.tensordot(resized, weights, [[axis], [-1]])
        resized = np.moveaxis(product, -1, axis)
    if channel_axis is not None:
        resized = np.moveaxis(resized, -1, channel_axis)
    return resized