import numpy as np
from ._util import _validate_connectivity, _offsets_to_raveled_neighbors
from ..util import invert
from . import _max_tree
def diameter_closing(image, diameter_threshold=8, connectivity=1, parent=None, tree_traverser=None):
    """Perform a diameter closing of the image.

    Diameter closing removes all dark structures of an image with
    maximal extension smaller than diameter_threshold. The maximal
    extension is defined as the maximal extension of the bounding box.
    The operator is also called Bounding Box Closing. In practice,
    the result is similar to a morphological closing, but long and thin
    structures are not removed.

    Technically, this operator is based on the max-tree representation of
    the image.

    Parameters
    ----------
    image : ndarray
        The input image for which the diameter_closing is to be calculated.
        This image can be of any type.
    diameter_threshold : unsigned int
        The maximal extension parameter (number of pixels). The default value
        is 8.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.
    parent : ndarray, int64, optional
        Precomputed parent image representing the max tree of the inverted
        image. This function is fast, if precomputed parent and tree_traverser
        are provided. See Note for further details.
    tree_traverser : 1D array, int64, optional
        Precomputed traverser, where the pixels are ordered such that every
        pixel is preceded by its parent (except for the root which has no
        parent). This function is fast, if precomputed parent and
        tree_traverser are provided. See Note for further details.

    Returns
    -------
    output : ndarray
        Output image of the same shape and type as input image.

    See Also
    --------
    skimage.morphology.area_opening
    skimage.morphology.area_closing
    skimage.morphology.diameter_opening
    skimage.morphology.max_tree

    References
    ----------
    .. [1] Walter, T., & Klein, J.-C. (2002). Automatic Detection of
           Microaneurysms in Color Fundus Images of the Human Retina by Means
           of the Bounding Box Closing. In A. Colosimo, P. Sirabella,
           A. Giuliani (Eds.), Medical Data Analysis. Lecture Notes in Computer
           Science, vol 2526, pp. 210-220. Springer Berlin Heidelberg.
           :DOI:`10.1007/3-540-36104-9_23`
    .. [2] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

    Examples
    --------
    We create an image (quadratic function with a minimum in the center and
    4 additional local minima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120
    >>> f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(int)

    We can calculate the diameter closing:

    >>> closed = diameter_closing(f, 3, connectivity=1)

    All small minima with a maximal extension of 2 or less are removed.
    The remaining minima have all a maximal extension of at least 3.

    Notes
    -----
    If a max-tree representation (parent and tree_traverser) are given to the
    function, they must be calculated from the inverted image for this
    function, i.e.:
    >>> P, S = max_tree(invert(f))
    >>> closed = diameter_closing(f, 3, parent=P, tree_traverser=S)
    """
    image_inv = invert(image)
    output = image_inv.copy()
    if parent is None or tree_traverser is None:
        parent, tree_traverser = max_tree(image_inv, connectivity)
    diam = _max_tree._compute_extension(image_inv.ravel(), np.array(image_inv.shape, dtype=np.int32), parent.ravel(), tree_traverser)
    _max_tree._direct_filter(image_inv.ravel(), output.ravel(), parent.ravel(), tree_traverser, diam, diameter_threshold)
    output = invert(output)
    return output