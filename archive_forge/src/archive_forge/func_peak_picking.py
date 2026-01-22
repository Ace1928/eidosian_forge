from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0, pre_max=1, post_max=1):
    """
    Perform thresholding and peak-picking on the given activation function.

    Parameters
    ----------
    activations : numpy array
        Activation function.
    threshold : float
        Threshold for peak-picking
    smooth : int or numpy array, optional
        Smooth the activation function with the kernel (size).
    pre_avg : int, optional
        Use `pre_avg` frames past information for moving average.
    post_avg : int, optional
        Use `post_avg` frames future information for moving average.
    pre_max : int, optional
        Use `pre_max` frames past information for moving maximum.
    post_max : int, optional
        Use `post_max` frames future information for moving maximum.

    Returns
    -------
    peak_idx : numpy array
        Indices of the detected peaks.

    See Also
    --------
    :func:`smooth`

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), set `pre_avg` and
    `post_avg` to 0.
    For peak picking of local maxima, set `pre_max` and  `post_max` to 1.
    For online peak picking, set all `post_` parameters to 0.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Markus Schedl,
           "Evaluating the Online Capabilities of Onset Detection Methods",
           Proceedings of the 13th International Society for Music Information
           Retrieval Conference (ISMIR), 2012.

    """
    activations = smooth_signal(activations, smooth)
    avg_length = pre_avg + post_avg + 1
    if avg_length > 1:
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        if activations.ndim == 1:
            filter_size = avg_length
        elif activations.ndim == 2:
            filter_size = [avg_length, 1]
        else:
            raise ValueError('`activations` must be either 1D or 2D')
        mov_avg = uniform_filter(activations, filter_size, mode='constant', origin=avg_origin)
    else:
        mov_avg = 0
    detections = activations * (activations >= mov_avg + threshold)
    max_length = pre_max + post_max + 1
    if max_length > 1:
        max_origin = int(np.floor((pre_max - post_max) / 2))
        if activations.ndim == 1:
            filter_size = max_length
        elif activations.ndim == 2:
            filter_size = [max_length, 1]
        else:
            raise ValueError('`activations` must be either 1D or 2D')
        mov_max = maximum_filter(detections, filter_size, mode='constant', origin=max_origin)
        detections *= detections == mov_max
    if activations.ndim == 1:
        return np.nonzero(detections)[0]
    elif activations.ndim == 2:
        return np.nonzero(detections)
    else:
        raise ValueError('`activations` must be either 1D or 2D')