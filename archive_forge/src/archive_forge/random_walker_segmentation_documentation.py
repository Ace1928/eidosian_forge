from *Random walks for image segmentation*, Leo Grady, IEEE Trans
import numpy as np
from scipy import sparse, ndimage as ndi
from .._shared import utils
from .._shared.utils import warn
from .._shared.compat import SCIPY_CG_TOL_PARAM_NAME
from ..util import img_as_float
from scipy.sparse.linalg import cg, spsolve
Random walker algorithm for segmentation from markers.

    Random walker algorithm is implemented for gray-level or multichannel
    images.

    Parameters
    ----------
    data : (M, N[, P][, C]) ndarray
        Image to be segmented in phases. Gray-level `data` can be two- or
        three-dimensional; multichannel data can be three- or four-
        dimensional with `channel_axis` specifying the dimension containing
        channels. Data spacing is assumed isotropic unless the `spacing`
        keyword argument is used.
    labels : (M, N[, P]) array of ints
        Array of seed markers labeled with different positive integers
        for different phases. Zero-labeled pixels are unlabeled pixels.
        Negative labels correspond to inactive pixels that are not taken
        into account (they are removed from the graph). If labels are not
        consecutive integers, the labels array will be transformed so that
        labels are consecutive. In the multichannel case, `labels` should have
        the same shape as a single channel of `data`, i.e. without the final
        dimension denoting channels.
    beta : float, optional
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).
    mode : string, available options {'cg', 'cg_j', 'cg_mg', 'bf'}
        Mode for solving the linear system in the random walker algorithm.

        - 'bf' (brute force): an LU factorization of the Laplacian is
          computed. This is fast for small images (<1024x1024), but very slow
          and memory-intensive for large images (e.g., 3-D volumes).
        - 'cg' (conjugate gradient): the linear system is solved iteratively
          using the Conjugate Gradient method from scipy.sparse.linalg. This is
          less memory-consuming than the brute force method for large images,
          but it is quite slow.
        - 'cg_j' (conjugate gradient with Jacobi preconditionner): the
          Jacobi preconditionner is applied during the Conjugate
          gradient method iterations. This may accelerate the
          convergence of the 'cg' method.
        - 'cg_mg' (conjugate gradient with multigrid preconditioner): a
          preconditioner is computed using a multigrid solver, then the
          solution is computed with the Conjugate Gradient method. This mode
          requires that the pyamg module is installed.
    tol : float, optional
        Tolerance to achieve when solving the linear system using
        the conjugate gradient based modes ('cg', 'cg_j' and 'cg_mg').
    copy : bool, optional
        If copy is False, the `labels` array will be overwritten with
        the result of the segmentation. Use copy=False if you want to
        save on memory.
    return_full_prob : bool, optional
        If True, the probability that a pixel belongs to each of the
        labels will be returned, instead of only the most likely
        label.
    spacing : iterable of floats, optional
        Spacing between voxels in each spatial dimension. If `None`, then
        the spacing between pixels/voxels in each dimension is assumed 1.
    prob_tol : float, optional
        Tolerance on the resulting probability to be in the interval [0, 1].
        If the tolerance is not satisfied, a warning is displayed.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    output : ndarray
        * If `return_full_prob` is False, array of ints of same shape
          and data type as `labels`, in which each pixel has been
          labeled according to the marker that reached the pixel first
          by anisotropic diffusion.
        * If `return_full_prob` is True, array of floats of shape
          `(nlabels, labels.shape)`. `output[label_nb, i, j]` is the
          probability that label `label_nb` reaches the pixel `(i, j)`
          first.

    See Also
    --------
    skimage.segmentation.watershed
        A segmentation algorithm based on mathematical morphology
        and "flooding" of regions from markers.

    Notes
    -----
    Multichannel inputs are scaled with all channel data combined. Ensure all
    channels are separately normalized prior to running this algorithm.

    The `spacing` argument is specifically for anisotropic datasets, where
    data points are spaced differently in one or more spatial dimensions.
    Anisotropic data is commonly encountered in medical imaging.

    The algorithm was first proposed in [1]_.

    The algorithm solves the diffusion equation at infinite times for
    sources placed on markers of each phase in turn. A pixel is labeled with
    the phase that has the greatest probability to diffuse first to the pixel.

    The diffusion equation is solved by minimizing x.T L x for each phase,
    where L is the Laplacian of the weighted graph of the image, and x is
    the probability that a marker of the given phase arrives first at a pixel
    by diffusion (x=1 on markers of the phase, x=0 on the other markers, and
    the other coefficients are looked for). Each pixel is attributed the label
    for which it has a maximal value of x. The Laplacian L of the image
    is defined as:

       - L_ii = d_i, the number of neighbors of pixel i (the degree of i)
       - L_ij = -w_ij if i and j are adjacent pixels

    The weight w_ij is a decreasing function of the norm of the local gradient.
    This ensures that diffusion is easier between pixels of similar values.

    When the Laplacian is decomposed into blocks of marked and unmarked
    pixels::

        L = M B.T
            B A

    with first indices corresponding to marked pixels, and then to unmarked
    pixels, minimizing x.T L x for one phase amount to solving::

        A x = - B x_m

    where x_m = 1 on markers of the given phase, and 0 on other markers.
    This linear system is solved in the algorithm using a direct method for
    small images, and an iterative method for larger images.

    References
    ----------
    .. [1] Leo Grady, Random walks for image segmentation, IEEE Trans Pattern
        Anal Mach Intell. 2006 Nov;28(11):1768-83.
        :DOI:`10.1109/TPAMI.2006.233`.

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> a = np.zeros((10, 10)) + 0.2 * rng.random((10, 10))
    >>> a[5:8, 5:8] += 1
    >>> b = np.zeros_like(a, dtype=np.int32)
    >>> b[3, 3] = 1  # Marker for first phase
    >>> b[6, 6] = 2  # Marker for second phase
    >>> random_walker(a, b)  # doctest: +SKIP
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)

    