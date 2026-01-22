import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
def compute_noise_components(imgseries, mask_images, components_criterion=0.5, filter_type=False, degree=0, period_cut=128, repetition_time=None, failure_mode='error', mask_names=None):
    """
    Compute the noise components from the image series for each mask.

    Parameters
    ----------
    imgseries: nibabel image
        Time series data to be decomposed.
    mask_images: list
        List of nibabel images. Time series data from ``img_series`` is subset
        according to the spatial extent of each mask, and the subset data is
        then decomposed using principal component analysis. Masks should be
        coextensive with either anatomical or spatial noise ROIs.
    components_criterion: float
        Number of noise components to return. If this is a decimal value
        between 0 and 1, then ``create_noise_components`` will instead return
        the smallest number of components necessary to explain the indicated
        fraction of variance. If ``components_criterion`` is ``all``, then all
        components will be returned.
    filter_type: str
        Type of filter to apply to time series before computing noise components.

            - 'polynomial' - Legendre polynomial basis
            - 'cosine' - Discrete cosine (DCT) basis
            - False - None (mean-removal only)

    failure_mode: str
        Action to be taken in the event that any decomposition fails to
        identify any components. ``error`` indicates that the routine should
        raise an exception and exit, while any other value indicates that the
        routine should return a matrix of NaN values equal in size to the
        requested decomposition matrix.
    mask_names: list or None
        List of names for each image in ``mask_images``. This should be equal in
        length to ``mask_images``, with the ith element of ``mask_names`` naming
        the ith element of ``mask_images``.
    degree: int
        Order of polynomial used to remove trends from the timeseries
    period_cut: float
        Minimum period (in sec) for DCT high-pass filter
    repetition_time: float
        Time (in sec) between volume acquisitions. This must be defined if
        the ``filter_type`` is ``cosine``.

    Returns
    -------
    components: numpy array
        Numpy array containing the requested set of noise components
    basis: numpy array
        Numpy array containing the (non-constant) filter regressors
    metadata: OrderedDict{str: numpy array}
        Dictionary of eigenvalues, fractional explained variances, and
        cumulative explained variances.

    """
    basis = np.array([])
    if components_criterion == 'all':
        components_criterion = -1
    mask_names = mask_names or range(len(mask_images))
    components = []
    md_mask = []
    md_sv = []
    md_var = []
    md_cumvar = []
    md_retained = []
    for name, img in zip(mask_names, mask_images):
        mask = np.asanyarray(nb.squeeze_image(img).dataobj).astype(bool)
        if imgseries.shape[:3] != mask.shape:
            raise ValueError('Inputs for CompCor, timeseries and mask, do not have matching spatial dimensions ({} and {}, respectively)'.format(imgseries.shape[:3], mask.shape))
        voxel_timecourses = imgseries[mask, :]
        voxel_timecourses[np.isnan(np.sum(voxel_timecourses, axis=1)), :] = 0
        if filter_type == 'cosine':
            if repetition_time is None:
                raise ValueError('Repetition time must be provided for cosine filter')
            voxel_timecourses, basis = cosine_filter(voxel_timecourses, repetition_time, period_cut, failure_mode=failure_mode)
        elif filter_type in ('polynomial', False):
            voxel_timecourses, basis = regress_poly(degree, voxel_timecourses, failure_mode=failure_mode)
        M = voxel_timecourses.T
        M = M / _compute_tSTD(M, 1.0)
        try:
            u, s, _ = fallback_svd(M, full_matrices=False)
        except (np.linalg.LinAlgError, ValueError):
            if failure_mode == 'error':
                raise
            s = np.full(M.shape[0], np.nan, dtype=np.float32)
            if components_criterion >= 1:
                u = np.full((M.shape[0], components_criterion), np.nan, dtype=np.float32)
            else:
                u = np.full((M.shape[0], 1), np.nan, dtype=np.float32)
        variance_explained = s ** 2 / np.sum(s ** 2)
        cumulative_variance_explained = np.cumsum(variance_explained)
        num_components = int(components_criterion)
        if 0 < components_criterion < 1:
            num_components = np.searchsorted(cumulative_variance_explained, components_criterion) + 1
        elif components_criterion == -1:
            num_components = len(s)
        num_components = int(num_components)
        if num_components == 0:
            break
        components.append(u[:, :num_components])
        md_mask.append([name] * len(s))
        md_sv.append(s)
        md_var.append(variance_explained)
        md_cumvar.append(cumulative_variance_explained)
        md_retained.append([i < num_components for i in range(len(s))])
    if len(components) > 0:
        components = np.hstack(components)
    else:
        if failure_mode == 'error':
            raise ValueError('No components found')
        components = np.full((M.shape[0], num_components), np.nan, dtype=np.float32)
    metadata = OrderedDict([('mask', list(chain(*md_mask))), ('singular_value', np.hstack(md_sv)), ('variance_explained', np.hstack(md_var)), ('cumulative_variance_explained', np.hstack(md_cumvar)), ('retained', list(chain(*md_retained)))])
    return (components, basis, metadata)