from functools import partial
import numpy as np
import scipy.fft as fftmodule
from scipy.fft import next_fast_len
from .._shared.utils import _supported_float_type
def cross_correlate_masked(arr1, arr2, m1, m2, mode='full', axes=(-2, -1), overlap_ratio=0.3):
    """
    Masked normalized cross-correlation between arrays.

    Parameters
    ----------
    arr1 : ndarray
        First array.
    arr2 : ndarray
        Seconds array. The dimensions of `arr2` along axes that are not
        transformed should be equal to that of `arr1`.
    m1 : ndarray
        Mask of `arr1`. The mask should evaluate to `True`
        (or 1) on valid pixels. `m1` should have the same shape as `arr1`.
    m2 : ndarray
        Mask of `arr2`. The mask should evaluate to `True`
        (or 1) on valid pixels. `m2` should have the same shape as `arr2`.
    mode : {'full', 'same'}, optional
        'full':
            This returns the convolution at each point of overlap. At
            the end-points of the convolution, the signals do not overlap
            completely, and boundary effects may be seen.
        'same':
            The output is the same size as `arr1`, centered with respect
            to the `‘full’` output. Boundary effects are less prominent.
    axes : tuple of ints, optional
        Axes along which to compute the cross-correlation.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images.

    Returns
    -------
    out : ndarray
        Masked normalized cross-correlation.

    Raises
    ------
    ValueError : if correlation `mode` is not valid, or array dimensions along
        non-transformation axes are not equal.

    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
    """
    if mode not in {'full', 'same'}:
        raise ValueError(f"Correlation mode '{mode}' is not valid.")
    fixed_image = np.asarray(arr1)
    moving_image = np.asarray(arr2)
    float_dtype = _supported_float_type((fixed_image.dtype, moving_image.dtype))
    if float_dtype.kind == 'c':
        raise ValueError('complex-valued arr1, arr2 are not supported')
    fixed_image = fixed_image.astype(float_dtype)
    fixed_mask = np.array(m1, dtype=bool)
    moving_image = moving_image.astype(float_dtype)
    moving_mask = np.array(m2, dtype=bool)
    eps = np.finfo(float_dtype).eps
    all_axes = set(range(fixed_image.ndim))
    for axis in all_axes - set(axes):
        if fixed_image.shape[axis] != moving_image.shape[axis]:
            raise ValueError(f'Array shapes along non-transformation axes should be equal, but dimensions along axis {axis} are not.')
    final_shape = list(arr1.shape)
    for axis in axes:
        final_shape[axis] = fixed_image.shape[axis] + moving_image.shape[axis] - 1
    final_shape = tuple(final_shape)
    final_slice = tuple([slice(0, int(sz)) for sz in final_shape])
    fast_shape = tuple([next_fast_len(final_shape[ax]) for ax in axes])
    fft = partial(fftmodule.fftn, s=fast_shape, axes=axes)
    _ifft = partial(fftmodule.ifftn, s=fast_shape, axes=axes)

    def ifft(x):
        return _ifft(x).real
    fixed_image[np.logical_not(fixed_mask)] = 0.0
    moving_image[np.logical_not(moving_mask)] = 0.0
    rotated_moving_image = _flip(moving_image, axes=axes)
    rotated_moving_mask = _flip(moving_mask, axes=axes)
    fixed_fft = fft(fixed_image)
    rotated_moving_fft = fft(rotated_moving_image)
    fixed_mask_fft = fft(fixed_mask.astype(float_dtype))
    rotated_moving_mask_fft = fft(rotated_moving_mask.astype(float_dtype))
    number_overlap_masked_px = ifft(rotated_moving_mask_fft * fixed_mask_fft)
    number_overlap_masked_px[:] = np.round(number_overlap_masked_px)
    number_overlap_masked_px[:] = np.fmax(number_overlap_masked_px, eps)
    masked_correlated_fixed_fft = ifft(rotated_moving_mask_fft * fixed_fft)
    masked_correlated_rotated_moving_fft = ifft(fixed_mask_fft * rotated_moving_fft)
    numerator = ifft(rotated_moving_fft * fixed_fft)
    numerator -= masked_correlated_fixed_fft * masked_correlated_rotated_moving_fft / number_overlap_masked_px
    fixed_squared_fft = fft(np.square(fixed_image))
    fixed_denom = ifft(rotated_moving_mask_fft * fixed_squared_fft)
    fixed_denom -= np.square(masked_correlated_fixed_fft) / number_overlap_masked_px
    fixed_denom[:] = np.fmax(fixed_denom, 0.0)
    rotated_moving_squared_fft = fft(np.square(rotated_moving_image))
    moving_denom = ifft(fixed_mask_fft * rotated_moving_squared_fft)
    moving_denom -= np.square(masked_correlated_rotated_moving_fft) / number_overlap_masked_px
    moving_denom[:] = np.fmax(moving_denom, 0.0)
    denom = np.sqrt(fixed_denom * moving_denom)
    numerator = numerator[final_slice]
    denom = denom[final_slice]
    number_overlap_masked_px = number_overlap_masked_px[final_slice]
    if mode == 'same':
        _centering = partial(_centered, newshape=fixed_image.shape, axes=axes)
        denom = _centering(denom)
        numerator = _centering(numerator)
        number_overlap_masked_px = _centering(number_overlap_masked_px)
    tol = 1000.0 * eps * np.max(np.abs(denom), axis=axes, keepdims=True)
    nonzero_indices = denom > tol
    out = np.zeros_like(denom, dtype=float_dtype)
    out[nonzero_indices] = numerator[nonzero_indices] / denom[nonzero_indices]
    np.clip(out, a_min=-1, a_max=1, out=out)
    number_px_threshold = overlap_ratio * np.max(number_overlap_masked_px, axis=axes, keepdims=True)
    out[number_overlap_masked_px < number_px_threshold] = 0.0
    return out