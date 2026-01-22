import hashlib
import os
import re
import sys
from collections import OrderedDict
from optparse import Option, OptionParser
import numpy as np
import nibabel as nib
import nibabel.cmdline.utils
def get_data_diff(files, max_abs=0, max_rel=0, dtype=np.float64):
    """Get difference between data

    Parameters
    ----------
    files: list of (str or ndarray)
      If list of strings is provided -- they must be existing file names
    max_abs: float, optional
      Maximal absolute difference to tolerate.
    max_rel: float, optional
      Maximal relative (`abs(diff)/mean(diff)`) difference to tolerate.
      If `max_abs` is specified, then those data points with lesser than that
      absolute difference, are not considered for relative difference testing
    dtype: np, optional
      Datatype to be used when extracting data from files

    Returns
    -------
    diffs: OrderedDict
        An ordered dict with a record per each file which has differences
        with other files subsequent detected. Each record is a list of
        difference records, one per each file pair.
        Each difference record is an Ordered Dict with possible keys
        'abs' or 'rel' showing maximal absolute or relative differences
        in the file or the record ('CMP': 'incompat') if file shapes
        are incompatible.
    """
    data = [f if isinstance(f, np.ndarray) else nib.load(f).get_fdata(dtype=dtype) for f in files]
    diffs = OrderedDict()
    for i, d1 in enumerate(data[:-1]):
        diffs1 = [None] * (i + 1)
        for j, d2 in enumerate(data[i + 1:], i + 1):
            if d1.shape == d2.shape:
                abs_diff = np.abs(d1 - d2)
                mean_abs = (np.abs(d1) + np.abs(d2)) * 0.5
                candidates = np.logical_or(mean_abs != 0, abs_diff != 0)
                if max_abs:
                    candidates[abs_diff <= max_abs] = False
                max_abs_diff = np.max(abs_diff)
                if np.any(candidates):
                    rel_diff = abs_diff[candidates] / mean_abs[candidates]
                    if max_rel:
                        sub_thr = rel_diff <= max_rel
                        candidates[tuple((indexes[sub_thr] for indexes in np.where(candidates)))] = False
                    max_rel_diff = np.max(rel_diff)
                else:
                    max_rel_diff = 0
                if np.any(candidates):
                    diff_rec = OrderedDict()
                    diff_rec['abs'] = max_abs_diff.astype(dtype)
                    diff_rec['rel'] = max_rel_diff.astype(dtype)
                    diffs1.append(diff_rec)
                else:
                    diffs1.append(None)
            else:
                diffs1.append({'CMP': 'incompat'})
        if any(diffs1):
            diffs['DATA(diff %d:)' % (i + 1)] = diffs1
    return diffs