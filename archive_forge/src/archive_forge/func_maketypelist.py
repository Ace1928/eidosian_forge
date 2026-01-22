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
def maketypelist(rowheadings, shape, extraheadingBool, extraheading):
    typelist = []
    if rowheadings:
        typelist.append(('heading', 'a40'))
    if len(shape) > 1:
        for idx in range(1, min(shape) + 1):
            typelist.append((str(idx), float))
    else:
        for idx in range(1, shape[0] + 1):
            typelist.append((str(idx), float))
    if extraheadingBool:
        typelist.append((extraheading, 'a40'))
    iflogger.info(typelist)
    return typelist