import os
from copy import deepcopy
from nibabel import load
import numpy as np
from ... import logging
from ...utils import spm_docs as sd
from ..base import (
from ..base.traits_extension import NoDefaultSpecified
from ..matlab import MatlabCommand
from ...external.due import due, Doi, BibTeX
def scans_for_fnames(fnames, keep4d=False, separate_sessions=False):
    """Converts a list of files to a concatenated numpy array for each
    volume.

    keep4d : boolean
        keeps the entries of the numpy array as 4d files instead of
        extracting the individual volumes.
    separate_sessions: boolean
        if 4d nifti files are being used, then separate_sessions
        ensures a cell array per session is created in the structure.

    """
    flist = None
    if not isinstance(fnames[0], list):
        if func_is_3d(fnames[0]):
            fnames = [fnames]
    if separate_sessions or keep4d:
        flist = np.zeros((len(fnames),), dtype=object)
    for i, f in enumerate(fnames):
        if separate_sessions:
            if keep4d:
                if isinstance(f, list):
                    flist[i] = np.array(f, dtype=object)
                else:
                    flist[i] = np.array([f], dtype=object)
            else:
                flist[i] = scans_for_fname(f)
        elif keep4d:
            flist[i] = f
        else:
            scans = scans_for_fname(f)
            if flist is None:
                flist = scans
            else:
                flist = np.concatenate((flist, scans))
    return flist