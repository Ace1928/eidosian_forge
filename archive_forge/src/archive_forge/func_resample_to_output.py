import numpy as np
import numpy.linalg as npl
from .optpkg import optional_package
from .affines import AffineError, append_diag, from_matvec, rescale_affine, to_matvec
from .imageclasses import spatial_axes_first
from .nifti1 import Nifti1Image
from .orientations import axcodes2ornt, io_orientation, ornt_transform
from .spaces import vox2out_vox
def resample_to_output(in_img, voxel_sizes=None, order=3, mode='constant', cval=0.0, out_class=Nifti1Image):
    """Resample image `in_img` to output voxel axes (world space)

    Parameters
    ----------
    in_img : object
        Object having attributes ``dataobj``, ``affine``, ``header``. If
        `out_class` is not None, ``img.__class__`` should be able to construct
        an image from data, affine and header.
    voxel_sizes : None or sequence
        Gives the diagonal entries of ``out_img.affine` (except the trailing 1
        for the homogeneous coordinates) (``out_img.affine ==
        np.diag(voxel_sizes + [1])``). If None, return identity
        `out_img.affine`.  If scalar, interpret as vector ``[voxel_sizes] *
        len(in_img.shape)``.
    order : int, optional
        The order of the spline interpolation, default is 3.  The order has to
        be in the range 0-5 (see ``scipy.ndimage.affine_transform``).
    mode : str, optional
        Points outside the boundaries of the input are filled according to the
        given mode ('constant', 'nearest', 'reflect' or 'wrap').  Default is
        'constant' (see ``scipy.ndimage.affine_transform``).
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0 (see
        ``scipy.ndimage.affine_transform``).
    out_class : None or SpatialImage class, optional
        Class of output image.  If None, use ``in_img.__class__``.

    Returns
    -------
    out_img : object
        Image of instance specified by `out_class`, containing data output from
        resampling `in_img` into axes aligned to the output space of
        ``in_img.affine``
    """
    if out_class is None:
        out_class = in_img.__class__
    in_shape = in_img.shape
    n_dim = len(in_shape)
    if voxel_sizes is not None:
        voxel_sizes = np.asarray(voxel_sizes)
        if voxel_sizes.ndim == 0:
            voxel_sizes = np.repeat(voxel_sizes, n_dim)
    if n_dim < 3:
        new_shape = in_shape + (1,) * (3 - n_dim)
        data = np.asanyarray(in_img.dataobj).reshape(new_shape)
        in_img = out_class(data, in_img.affine, in_img.header)
        if voxel_sizes is not None and len(voxel_sizes) == n_dim:
            voxel_sizes = tuple(voxel_sizes) + (1,) * (3 - n_dim)
    out_vox_map = vox2out_vox((in_img.shape, in_img.affine), voxel_sizes)
    return resample_from_to(in_img, out_vox_map, order, mode, cval, out_class)