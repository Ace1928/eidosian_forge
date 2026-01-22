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
def split_rois(in_file, mask=None, roishape=None):
    """
    Splits an image in ROIs for parallel processing
    """
    import nibabel as nb
    import numpy as np
    from math import sqrt, ceil
    import os.path as op
    if roishape is None:
        roishape = (10, 10, 1)
    im = nb.load(in_file)
    imshape = im.shape
    dshape = imshape[:3]
    nvols = imshape[-1]
    roisize = roishape[0] * roishape[1] * roishape[2]
    droishape = (roishape[0], roishape[1], roishape[2], nvols)
    if mask is not None:
        mask = np.asanyarray(nb.load(mask).dataobj)
        mask[mask > 0] = 1
        mask[mask < 1] = 0
    else:
        mask = np.ones(dshape)
    mask = mask.reshape(-1).astype(np.uint8)
    nzels = np.nonzero(mask)
    els = np.sum(mask)
    nrois = int(ceil(els / float(roisize)))
    data = np.asanyarray(im.dataobj).reshape((mask.size, -1))
    data = np.squeeze(data.take(nzels, axis=0))
    nvols = data.shape[-1]
    roidefname = op.abspath('onesmask.nii.gz')
    nb.Nifti1Image(np.ones(roishape, dtype=np.uint8), None, None).to_filename(roidefname)
    out_files = []
    out_mask = []
    out_idxs = []
    for i in range(nrois):
        first = i * roisize
        last = (i + 1) * roisize
        fill = 0
        if last > els:
            fill = last - els
            last = els
        droi = data[first:last, ...]
        iname = op.abspath('roi%010d_idx' % i)
        out_idxs.append(iname + '.npz')
        np.savez(iname, (nzels[0][first:last],))
        if fill > 0:
            droi = np.vstack((droi, np.zeros((int(fill), int(nvols)), dtype=np.float32)))
            partialmsk = np.ones((roisize,), dtype=np.uint8)
            partialmsk[-int(fill):] = 0
            partname = op.abspath('partialmask.nii.gz')
            nb.Nifti1Image(partialmsk.reshape(roishape), None, None).to_filename(partname)
            out_mask.append(partname)
        else:
            out_mask.append(roidefname)
        fname = op.abspath('roi%010d.nii.gz' % i)
        nb.Nifti1Image(droi.reshape(droishape), None, None).to_filename(fname)
        out_files.append(fname)
    return (out_files, out_mask, out_idxs)