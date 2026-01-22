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
def get_test_label(test):
    have_igzip = test[0]
    keep_open = test[1]
    if not (have_igzip and keep_open):
        return 'gzip'
    else:
        return 'indexed_gzip'