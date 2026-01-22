import numpy as np
from .loadsave import load
from .orientations import OrientationError, io_orientation
def squeeze_image(img):
    """Return image, remove axes length 1 at end of image shape

    For example, an image may have shape (10,20,30,1,1).  In this case
    squeeze will result in an image with shape (10,20,30).  See doctests
    for further description of behavior.

    Parameters
    ----------
    img : ``SpatialImage``

    Returns
    -------
    squeezed_img : ``SpatialImage``
       Copy of img, such that data, and data shape have been squeezed,
       for dimensions > 3rd, and at the end of the shape list

    Examples
    --------
    >>> import nibabel as nf
    >>> shape = (10,20,30,1,1)
    >>> data = np.arange(np.prod(shape), dtype='int32').reshape(shape)
    >>> affine = np.eye(4)
    >>> img = nf.Nifti1Image(data, affine)
    >>> img.shape == (10, 20, 30, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (10, 20, 30)
    True

    If the data are 3D then last dimensions of 1 are ignored

    >>> shape = (10,1,1)
    >>> data = np.arange(np.prod(shape), dtype='int32').reshape(shape)
    >>> img = nf.ni1.Nifti1Image(data, affine)
    >>> img.shape == (10, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (10, 1, 1)
    True

    Only *final* dimensions of 1 are squeezed

    >>> shape = (1, 1, 5, 1, 2, 1, 1)
    >>> data = data.reshape(shape)
    >>> img = nf.ni1.Nifti1Image(data, affine)
    >>> img.shape == (1, 1, 5, 1, 2, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (1, 1, 5, 1, 2)
    True
    """
    klass = img.__class__
    shape = img.shape
    slen = len(shape)
    if slen < 4:
        return klass.from_image(img)
    for bdim in shape[3:][::-1]:
        if bdim == 1:
            slen -= 1
        else:
            break
    if slen == len(shape):
        return klass.from_image(img)
    shape = shape[:slen]
    data = np.asanyarray(img.dataobj).reshape(shape)
    return klass(data, img.affine, img.header, img.extra)