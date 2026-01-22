import itertools
import numpy as np
from .._shared.utils import _supported_float_type, check_nD
from . import _moments_cy
from ._moments_analytical import moments_raw_to_central
def moments_normalized(mu, order=3, spacing=None):
    """Calculate all normalized central image moments up to a certain order.

    Note that normalized central moments are translation and scale invariant
    but not rotation invariant.

    Parameters
    ----------
    mu : (M[, ...], M) array
        Central image moments, where M must be greater than or equal
        to ``order``.
    order : int, optional
        Maximum order of moments. Default is 3.
    spacing: tuple of float, shape (ndim,)
        The pixel spacing along each axis of the image.

    Returns
    -------
    nu : (``order + 1``[, ...], ``order + 1``) array
        Normalized central image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.float64)
    >>> image[13:17, 13:17] = 1
    >>> m = moments(image)
    >>> centroid = (m[0, 1] / m[0, 0], m[1, 0] / m[0, 0])
    >>> mu = moments_central(image, centroid)
    >>> moments_normalized(mu)
    array([[       nan,        nan, 0.078125  , 0.        ],
           [       nan, 0.        , 0.        , 0.        ],
           [0.078125  , 0.        , 0.00610352, 0.        ],
           [0.        , 0.        , 0.        , 0.        ]])
    """
    if np.any(np.array(mu.shape) <= order):
        raise ValueError('Shape of image moments must be >= `order`')
    if spacing is None:
        spacing = np.ones(mu.ndim)
    nu = np.zeros_like(mu)
    mu0 = mu.ravel()[0]
    scale = min(spacing)
    for powers in itertools.product(range(order + 1), repeat=mu.ndim):
        if sum(powers) < 2:
            nu[powers] = np.nan
        else:
            nu[powers] = mu[powers] / scale ** sum(powers) / mu0 ** (sum(powers) / nu.ndim + 1)
    return nu