import os
from collections import OrderedDict
from os.path import join as pjoin, dirname
from glob import glob
from io import BytesIO
import re
from tempfile import mkdtemp
import warnings
import shutil
import gzip
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import array
import scipy.sparse as SP
import scipy.io
from scipy.io.matlab import MatlabOpaque, MatlabFunction, MatlabObject
import scipy.io.matlab._byteordercodes as boc
from scipy.io.matlab._miobase import (
from scipy.io.matlab._mio import mat_reader_factory, loadmat, savemat, whosmat
from scipy.io.matlab._mio5 import (
import scipy.io.matlab._mio5_params as mio5p
from scipy._lib._util import VisibleDeprecationWarning
def test_varmats_from_mat():
    names_vars = (('arr', mlarr(np.arange(10))), ('mystr', mlarr('a string')), ('mynum', mlarr(10)))

    class C:

        def items(self):
            return names_vars
    stream = BytesIO()
    savemat(stream, C())
    varmats = varmats_from_mat(stream)
    assert_equal(len(varmats), 3)
    for i in range(3):
        name, var_stream = varmats[i]
        exp_name, exp_res = names_vars[i]
        assert_equal(name, exp_name)
        res = loadmat(var_stream)
        assert_array_equal(res[name], exp_res)