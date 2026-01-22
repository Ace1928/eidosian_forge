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
def remove_identical_paths(in_files):
    import os.path as op
    from ..utils.filemanip import split_filename
    if len(in_files) > 1:
        out_names = list()
        commonprefix = op.commonprefix(in_files)
        lastslash = commonprefix.rfind('/')
        commonpath = commonprefix[0:lastslash + 1]
        for fileidx, in_file in enumerate(in_files):
            path, name, ext = split_filename(in_file)
            in_file = op.join(path, name)
            name = in_file.replace(commonpath, '')
            name = name.replace('_subject_id_', '')
            out_names.append(name)
    else:
        path, name, ext = split_filename(in_files[0])
        out_names = [name]
    return out_names