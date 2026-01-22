import numpy as np
import pytest
from skimage import io
from skimage._shared._warnings import expected_warnings
def n_subplots(ax_im):
    """Return the number of subplots in the figure containing an ``AxesImage``.

    Parameters
    ----------
    ax_im : matplotlib.pyplot.AxesImage object
        The input ``AxesImage``.

    Returns
    -------
    n : int
        The number of subplots in the corresponding figure.

    Notes
    -----
    This function is intended to check whether a colorbar was drawn, in
    which case two subplots are expected. For standard imshows, one
    subplot is expected.
    """
    return len(ax_im.get_figure().get_axes())