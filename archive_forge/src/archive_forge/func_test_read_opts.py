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
def test_read_opts():
    arr = np.arange(6).reshape(1, 6)
    stream = BytesIO()
    savemat(stream, {'a': arr})
    rdr = MatFile5Reader(stream)
    back_dict = rdr.get_variables()
    rarr = back_dict['a']
    assert_array_equal(rarr, arr)
    rdr = MatFile5Reader(stream, squeeze_me=True)
    assert_array_equal(rdr.get_variables()['a'], arr.reshape((6,)))
    rdr.squeeze_me = False
    assert_array_equal(rarr, arr)
    rdr = MatFile5Reader(stream, byte_order=boc.native_code)
    assert_array_equal(rdr.get_variables()['a'], arr)
    rdr = MatFile5Reader(stream, byte_order=boc.swapped_code)
    assert_raises(Exception, rdr.get_variables)
    rdr.byte_order = boc.native_code
    assert_array_equal(rdr.get_variables()['a'], arr)
    arr = np.array(['a string'])
    stream.truncate(0)
    stream.seek(0)
    savemat(stream, {'a': arr})
    rdr = MatFile5Reader(stream)
    assert_array_equal(rdr.get_variables()['a'], arr)
    rdr = MatFile5Reader(stream, chars_as_strings=False)
    carr = np.atleast_2d(np.array(list(arr.item()), dtype='U1'))
    assert_array_equal(rdr.get_variables()['a'], carr)
    rdr.chars_as_strings = True
    assert_array_equal(rdr.get_variables()['a'], arr)