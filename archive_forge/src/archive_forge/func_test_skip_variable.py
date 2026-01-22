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
def test_skip_variable():
    filename = pjoin(test_data_path, 'test_skip_variable.mat')
    d = loadmat(filename, struct_as_record=True)
    assert_('first' in d)
    assert_('second' in d)
    factory, file_opened = mat_reader_factory(filename, struct_as_record=True)
    d = factory.get_variables('second')
    assert_('second' in d)
    factory.mat_stream.close()