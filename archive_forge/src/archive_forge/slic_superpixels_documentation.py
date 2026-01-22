import math
from collections.abc import Iterable
from warnings import warn
import numpy as np
from numpy import random
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform
from .._shared import utils
from .._shared.filters import gaussian
from ..color import rgb2lab
from ..util import img_as_float, regular_grid
from ._slic import _enforce_label_connectivity_cython, _slic_cython
Segments image using k-means clustering in Color-(x,y,z) space.

    Parameters
    ----------
    image : (M, N[, P][, C]) ndarray
        Input image. Can be 2D or 3D, and grayscale or multichannel
        (see `channel_axis` parameter).
        Input image must either be NaN-free or the NaN's must be masked out.
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic. In SLICO mode, this is the initial compactness.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    max_num_iter : int, optional
        Maximum number of iterations of k-means.
    sigma : float or array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
        Note that `sigma` is automatically scaled if it is scalar and
        if a manual voxel spacing is provided (see Notes section). If
        sigma is array-like, its size must match ``image``'s number
        of spatial dimensions.
    spacing : array-like of floats, optional
        The voxel spacing along each spatial dimension. By default,
        `slic` assumes uniform spacing (same voxel resolution along
        each spatial dimension).
        This parameter controls the weights of the distances along the
        spatial dimensions during k-means clustering.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. The input image *must* be RGB. Highly recommended.
        This option defaults to ``True`` when ``channel_axis` is not None *and*
        ``image.shape[-1] == 3``.
    enforce_connectivity : bool, optional
        Whether the generated segments are connected or not
    min_size_factor : float, optional
        Proportion of the minimum segment size to be removed with respect
        to the supposed segment size ```depth*width*height/n_segments```
    max_size_factor : float, optional
        Proportion of the maximum connected segment size. A value of 3 works
        in most of the cases.
    slic_zero : bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. [2]_
    start_label : int, optional
        The labels' index start. Should be 0 or 1.

        .. versionadded:: 0.17
           ``start_label`` was introduced in 0.17
    mask : ndarray, optional
        If provided, superpixels are computed only where mask is True,
        and seed points are homogeneously distributed over the mask
        using a k-means clustering strategy. Mask number of dimensions
        must be equal to image number of spatial dimensions.

        .. versionadded:: 0.17
           ``mask`` was introduced in 0.17
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    labels : 2D or 3D array
        Integer mask indicating segment labels.

    Raises
    ------
    ValueError
        If ``convert2lab`` is set to ``True`` but the last array
        dimension is not of length 3.
    ValueError
        If ``start_label`` is not 0 or 1.
    ValueError
        If ``image`` contains unmasked NaN values.
    ValueError
        If ``image`` contains unmasked infinite values.
    ValueError
        If ``image`` is 2D but ``channel_axis`` is -1 (the default).

    Notes
    -----
    * If `sigma > 0`, the image is smoothed using a Gaussian kernel prior to
      segmentation.

    * If `sigma` is scalar and `spacing` is provided, the kernel width is
      divided along each dimension by the spacing. For example, if ``sigma=1``
      and ``spacing=[5, 1, 1]``, the effective `sigma` is ``[0.2, 1, 1]``. This
      ensures sensible smoothing for anisotropic images.

    * The image is rescaled to be in [0, 1] prior to processing (masked
      values are ignored).

    * Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
      interpret them as 3D with the last dimension having length 3, use
      `channel_axis=None`.

    * `start_label` is introduced to handle the issue [4]_. Label indexing
      starts at 1 by default.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
        :DOI:`10.1109/TPAMI.2012.120`
    .. [2] https://www.epfl.ch/labs/ivrl/research/slic-superpixels/#SLICO
    .. [3] Irving, Benjamin. "maskSLIC: regional superpixel generation with
           application to local pathology characterisation in medical images.",
           2016, :arXiv:`1606.09518`
    .. [4] https://github.com/scikit-image/scikit-image/issues/3722

    Examples
    --------
    >>> from skimage.segmentation import slic
    >>> from skimage.data import astronaut
    >>> img = astronaut()
    >>> segments = slic(img, n_segments=100, compactness=10)

    Increasing the compactness parameter yields more square regions:

    >>> segments = slic(img, n_segments=100, compactness=20)

    