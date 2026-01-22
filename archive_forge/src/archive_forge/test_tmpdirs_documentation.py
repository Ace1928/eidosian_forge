from os import getcwd
from os.path import realpath, abspath, dirname, isfile, join as pjoin, exists
from scipy._lib._tmpdirs import tempdir, in_tempdir, in_dir
from numpy.testing import assert_, assert_equal
 Test tmpdirs module 