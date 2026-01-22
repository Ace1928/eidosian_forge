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
def test_loadmat_varnames():
    mat5_sys_names = ['__globals__', '__header__', '__version__']
    for eg_file, sys_v_names in ((pjoin(test_data_path, 'testmulti_4.2c_SOL2.mat'), []), (pjoin(test_data_path, 'testmulti_7.4_GLNX86.mat'), mat5_sys_names)):
        vars = loadmat(eg_file)
        assert_equal(set(vars.keys()), set(['a', 'theta'] + sys_v_names))
        vars = loadmat(eg_file, variable_names='a')
        assert_equal(set(vars.keys()), set(['a'] + sys_v_names))
        vars = loadmat(eg_file, variable_names=['a'])
        assert_equal(set(vars.keys()), set(['a'] + sys_v_names))
        vars = loadmat(eg_file, variable_names=['theta'])
        assert_equal(set(vars.keys()), set(['theta'] + sys_v_names))
        vars = loadmat(eg_file, variable_names=('theta',))
        assert_equal(set(vars.keys()), set(['theta'] + sys_v_names))
        vars = loadmat(eg_file, variable_names=[])
        assert_equal(set(vars.keys()), set(sys_v_names))
        vnames = ['theta']
        vars = loadmat(eg_file, variable_names=vnames)
        assert_equal(vnames, ['theta'])