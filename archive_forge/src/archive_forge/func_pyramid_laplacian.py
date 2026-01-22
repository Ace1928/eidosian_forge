import math
import numpy as np
from .._shared.filters import gaussian
from .._shared.utils import convert_to_float
from ._warps import resize
def pyramid_laplacian(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None):
    """Yield images of the laplacian pyramid formed by the input image.

    Each layer contains the difference between the downsampled and the
    downsampled, smoothed image::

        layer = resize(prev_layer) - smooth(resize(prev_layer))

    Note that the first image of the pyramid will be the difference between the
    original, unscaled image and its smoothed version. The total number of
    images is `max_layer + 1`. In case all layers are computed, the last image
    is either a one-pixel image or the image where the reduction does not
    change its shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int, optional
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf
    .. [2] http://sepwww.stanford.edu/data/media/public/sep/morgan/texturematch/paper_html/node3.html

    """
    _check_factor(downscale)
    image = convert_to_float(image, preserve_range)
    if sigma is None:
        sigma = 2 * downscale / 6.0
    current_shape = image.shape
    smoothed_image = _smooth(image, sigma, mode, cval, channel_axis)
    yield (image - smoothed_image)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        shape_without_channels = list(current_shape)
        shape_without_channels.pop(channel_axis)
        shape_without_channels = tuple(shape_without_channels)
    else:
        shape_without_channels = current_shape
    if max_layer == -1:
        max_layer = math.ceil(math.log(max(shape_without_channels), downscale))
    for layer in range(max_layer):
        if channel_axis is not None:
            out_shape = tuple((math.ceil(d / float(downscale)) if ax != channel_axis else d for ax, d in enumerate(current_shape)))
        else:
            out_shape = tuple((math.ceil(d / float(downscale)) for d in current_shape))
        resized_image = resize(smoothed_image, out_shape, order=order, mode=mode, cval=cval, anti_aliasing=False)
        smoothed_image = _smooth(resized_image, sigma, mode, cval, channel_axis)
        current_shape = resized_image.shape
        yield (resized_image - smoothed_image)