import datetime
import os
import sys
from os.path import join as pjoin
from io import StringIO
import numpy as np
from numpy.testing import (assert_array_almost_equal,
from pytest import raises as assert_raises
from scipy.io.arff import loadarff
from scipy.io.arff._arffread import read_header, ParseArffError
def test_type_parsing(self):
    with open(test2) as ofile:
        rel, attrs = read_header(ofile)
    expected = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'string', 'string', 'nominal', 'nominal']
    for i in range(len(attrs)):
        assert_(attrs[i].type_name == expected[i])