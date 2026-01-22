import gc
import itertools as it
from timeit import timeit
from unittest import mock
import numpy as np
import nibabel as nib
from nibabel.openers import HAVE_INDEXED_GZIP
from nibabel.tmpdirs import InTemporaryDirectory
from ..rstutils import rst_table
from .butils import print_git_title
def fix_sliceobj(sliceobj):
    new_sliceobj = []
    for i, s in enumerate(sliceobj):
        if s == ':':
            new_sliceobj.append(slice(None))
        elif s == '?':
            new_sliceobj.append(np.random.randint(0, SHAPE[i]))
        else:
            new_sliceobj.append(int(s * SHAPE[i]))
    return tuple(new_sliceobj)