import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
def merge_rois(in_files, in_idxs, in_ref, dtype=None, out_file=None):
    """
    Re-builds an image resulting from a parallelized processing
    """
    import nibabel as nb
    import numpy as np
    import os.path as op
    import subprocess as sp
    if out_file is None:
        out_file = op.abspath('merged.nii.gz')
    if dtype is None:
        dtype = np.float32
    if op.splitext(in_ref)[1] == '.gz':
        try:
            iflogger.info('uncompress %s', in_ref)
            sp.check_call(['gunzip', in_ref], stdout=sp.PIPE, shell=True)
            in_ref = op.splitext(in_ref)[0]
        except:
            pass
    ref = nb.load(in_ref)
    aff = ref.affine
    hdr = ref.header.copy()
    rsh = ref.shape
    del ref
    npix = rsh[0] * rsh[1] * rsh[2]
    fcimg = nb.load(in_files[0])
    if len(fcimg.shape) == 4:
        ndirs = fcimg.shape[-1]
    else:
        ndirs = 1
    newshape = (rsh[0], rsh[1], rsh[2], ndirs)
    hdr.set_data_dtype(dtype)
    hdr.set_xyzt_units('mm', 'sec')
    if ndirs < 300:
        data = np.zeros((npix, ndirs), dtype=dtype)
        for cname, iname in zip(in_files, in_idxs):
            f = np.load(iname)
            idxs = np.squeeze(f['arr_0'])
            cdata = np.asanyarray(nb.load(cname).dataobj).reshape(-1, ndirs)
            nels = len(idxs)
            idata = (idxs,)
            try:
                data[idata, ...] = cdata[0:nels, ...]
            except:
                print('Consistency between indexes and chunks was lost: data=%s, chunk=%s' % (str(data.shape), str(cdata.shape)))
                raise
        nb.Nifti1Image(data.reshape(newshape), aff, hdr).to_filename(out_file)
    else:
        hdr.set_data_shape(rsh[:3])
        nii = []
        for d in range(ndirs):
            fname = op.abspath('vol%06d.nii' % d)
            nb.Nifti1Image(np.zeros(rsh[:3]), aff, hdr).to_filename(fname)
            nii.append(fname)
        for cname, iname in zip(in_files, in_idxs):
            f = np.load(iname)
            idxs = np.squeeze(f['arr_0'])
            for d, fname in enumerate(nii):
                data = np.asanyarray(nb.load(fname).dataobj).reshape(-1)
                cdata = nb.load(cname).dataobj[..., d].reshape(-1)
                nels = len(idxs)
                idata = (idxs,)
                data[idata] = cdata[0:nels]
                nb.Nifti1Image(data.reshape(rsh[:3]), aff, hdr).to_filename(fname)
        imgs = [nb.load(im) for im in nii]
        allim = nb.concat_images(imgs)
        allim.to_filename(out_file)
    return out_file