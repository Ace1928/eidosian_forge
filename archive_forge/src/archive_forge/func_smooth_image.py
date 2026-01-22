import numpy as np
import numpy.linalg as npl
from .optpkg import optional_package
from .affines import AffineError, append_diag, from_matvec, rescale_affine, to_matvec
from .imageclasses import spatial_axes_first
from .nifti1 import Nifti1Image
from .orientations import axcodes2ornt, io_orientation, ornt_transform
from .spaces import vox2out_vox
def smooth_image(img, fwhm, mode='nearest', cval=0.0, out_class=Nifti1Image):
    """Smooth image `img` along voxel axes by FWHM `fwhm` millimeters

    Parameters
    ----------
    img : object
        Object having attributes ``dataobj``, ``affine``, ``header`` and
        ``shape``. If `out_class` is not None, ``img.__class__`` should be able
        to construct an image from data, affine and header.
    fwhm : scalar or length 3 sequence
        FWHM *in mm* over which to smooth.  The smoothing applies to the voxel
        axes, not to the output axes, but is in millimeters.  The function
        adjusts the FWHM to voxels using the voxel sizes calculated from the
        affine. A scalar implies the same smoothing across the spatial
        dimensions of the image, but 0 smoothing over any further dimensions
        such as time.  A vector should be the same length as the number of
        image dimensions.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'nearest'. This is different from the default for
        ``scipy.ndimage.affine_transform``, which is 'constant'. 'nearest'
        might be a better choice when smoothing to the edge of an image where
        there is still strong brain signal, otherwise this signal will get
        blurred towards zero.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0 (see
        ``scipy.ndimage.affine_transform``).
    out_class : None or SpatialImage class, optional
        Class of output image.  If None, use ``img.__class__``.

    Returns
    -------
    smoothed_img : object
        Image of instance specified by `out_class`, containing data output from
        smoothing `img` data by given FWHM kernel.
    """
    if not spatial_axes_first(img):
        raise ValueError(f'Cannot predict position of spatial axes for Image type {type(img)}')
    if out_class is None:
        out_class = img.__class__
    n_dim = len(img.shape)
    fwhm = np.asarray(fwhm)
    if fwhm.size == 1:
        fwhm_scalar = fwhm
        fwhm = np.zeros((n_dim,))
        fwhm[:3] = fwhm_scalar
    RZS = img.affine[:, :n_dim]
    vox = np.sqrt(np.sum(RZS ** 2, 0))
    vox_fwhm = fwhm / vox
    vox_sd = fwhm2sigma(vox_fwhm)
    sm_data = spnd.gaussian_filter(img.dataobj, vox_sd, mode=mode, cval=cval)
    return out_class(sm_data, img.affine, img.header)