import os
import pytest
import numpy as np
from . import util
def test_include_path():
    incdir = np.f2py.get_include()
    fnames_in_dir = os.listdir(incdir)
    for fname in ('fortranobject.c', 'fortranobject.h'):
        assert fname in fnames_in_dir