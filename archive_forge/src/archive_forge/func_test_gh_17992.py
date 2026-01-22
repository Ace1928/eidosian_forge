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
def test_gh_17992(tmp_path):
    rng = np.random.default_rng(12345)
    outfile = tmp_path / 'lists.mat'
    array_one = rng.random((5, 3))
    array_two = rng.random((6, 3))
    list_of_arrays = [array_one, array_two]
    with np.testing.suppress_warnings() as sup:
        sup.filter(VisibleDeprecationWarning)
        savemat(outfile, {'data': list_of_arrays}, long_field_names=True, do_compression=True)
    new_dict = {}
    loadmat(outfile, new_dict)
    assert_allclose(new_dict['data'][0][0], array_one)
    assert_allclose(new_dict['data'][0][1], array_two)