from os.path import dirname, join as pjoin
from numpy.testing import assert_
from pytest import raises as assert_raises
from scipy.io.matlab._mio import loadmat
 Test reading of files not conforming to matlab specification

We try and read any file that matlab reads, these files included
