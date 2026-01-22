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
def test_use_small_element():
    sio = BytesIO()
    wtr = MatFile5Writer(sio)
    arr = np.zeros(10)
    wtr.put_variables({'aaaaa': arr})
    w_sz = len(sio.getvalue())
    sio.truncate(0)
    sio.seek(0)
    wtr.put_variables({'aaaa': arr})
    assert_(w_sz - len(sio.getvalue()) > 4)
    sio.truncate(0)
    sio.seek(0)
    wtr.put_variables({'aaaaaa': arr})
    assert_(len(sio.getvalue()) - w_sz < 4)