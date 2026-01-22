import numpy as np
import numpy.linalg as npl
from .optpkg import optional_package
from .affines import AffineError, append_diag, from_matvec, rescale_affine, to_matvec
from .imageclasses import spatial_axes_first
from .nifti1 import Nifti1Image
from .orientations import axcodes2ornt, io_orientation, ornt_transform
from .spaces import vox2out_vox
def resample_from_to(from_img, to_vox_map, order=3, mode='constant', cval=0.0, out_class=Nifti1Image):
    """Resample image `from_img` to mapped voxel space `to_vox_map`

    Resample using N-d spline interpolation.

    Parameters
    ----------
    from_img : object
        Object having attributes ``dataobj``, ``affine``, ``header`` and
        ``shape``. If `out_class` is not None, ``img.__class__`` should be able
        to construct an image from data, affine and header.
    to_vox_map : image object or length 2 sequence
        If object, has attributes ``shape`` giving input voxel shape, and
        ``affine`` giving mapping of input voxels to output space. If length 2
        sequence, elements are (shape, affine) with same meaning as above. The
        affine is a (4, 4) array-like.
    order : int, optional
        The order of the spline interpolation, default is 3.  The order has to
        be in the range 0-5 (see ``scipy.ndimage.affine_transform``)
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'constant' (see ``scipy.ndimage.affine_transform``)
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0 (see
        ``scipy.ndimage.affine_transform``)
    out_class : None or SpatialImage class, optional
        Class of output image.  If None, use ``from_img.__class__``.

    Returns
    -------
    out_img : object
        Image of instance specified by `out_class`, containing data output from
        resampling `from_img` into axes aligned to the output space of
        ``from_img.affine``
    """
    if not spatial_axes_first(from_img):
        raise ValueError(f'Cannot predict position of spatial axes for Image type {type(from_img)}')
    try:
        to_shape, to_affine = (to_vox_map.shape, to_vox_map.affine)
    except AttributeError:
        to_shape, to_affine = to_vox_map
    a_to_affine = adapt_affine(to_affine, len(to_shape))
    if out_class is None:
        out_class = from_img.__class__
    from_n_dim = len(from_img.shape)
    if from_n_dim < 3:
        raise AffineError('from_img must be at least 3D')
    a_from_affine = adapt_affine(from_img.affine, from_n_dim)
    to_vox2from_vox = npl.inv(a_from_affine).dot(a_to_affine)
    rzs, trans = to_matvec(to_vox2from_vox)
    data = spnd.affine_transform(from_img.dataobj, rzs, trans, to_shape, order=order, mode=mode, cval=cval)
    return out_class(data, to_affine, from_img.header)