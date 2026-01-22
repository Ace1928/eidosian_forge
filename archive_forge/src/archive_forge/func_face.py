from scipy._lib.deprecation import _deprecated
from scipy._lib._finite_differences import _central_diff_weights, _derivative
from numpy import array, frombuffer, load
@_deprecated(msg='scipy.misc.face has been deprecated in SciPy v1.10.0; and will be completely removed in SciPy v1.12.0. Dataset methods have moved into the scipy.datasets module. Use scipy.datasets.face instead.')
def face(gray=False):
    """
    Get a 1024 x 768, color image of a raccoon face.

    raccoon-procyon-lotor.jpg at http://www.public-domain-image.com

    .. deprecated:: 1.10.0
        `face` has been deprecated from `scipy.misc.face`
        in SciPy 1.10.0 and it will be completely removed in SciPy 1.12.0.
        Dataset methods have moved into the `scipy.datasets` module.
        Use `scipy.datasets.face` instead.

    Parameters
    ----------
    gray : bool, optional
        If True return 8-bit grey-scale image, otherwise return a color image

    Returns
    -------
    face : ndarray
        image of a raccoon face

    Examples
    --------
    >>> import scipy.misc
    >>> face = scipy.misc.face()
    >>> face.shape
    (768, 1024, 3)
    >>> face.max()
    255
    >>> face.dtype
    dtype('uint8')

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(face)
    >>> plt.show()

    """
    import bz2
    import os
    with open(os.path.join(os.path.dirname(__file__), 'face.dat'), 'rb') as f:
        rawdata = f.read()
    data = bz2.decompress(rawdata)
    face = frombuffer(data, dtype='uint8')
    face.shape = (768, 1024, 3)
    if gray is True:
        face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] + 0.07 * face[:, :, 2]).astype('uint8')
    return face