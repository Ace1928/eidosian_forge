import itertools
import functools
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..metrics import mean_squared_error
from ..util import img_as_float
Return a parameter search history with losses for a denoise function.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Denoising function to be calibrated.
    denoise_parameters : dict of list
        Ranges of parameters for `denoise_function` to be calibrated over.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    approximate_loss : bool, optional
        Whether to approximate the self-supervised loss used to evaluate the
        denoiser by only computing it on one masked version of the image.
        If False, the runtime will be a factor of `stride**image.ndim` longer.

    Returns
    -------
    parameters_tested : list of dict
        List of parameters tested for `denoise_function`, as a dictionary of
        kwargs.
    losses : list of int
        Self-supervised loss for each set of parameters in `parameters_tested`.
    