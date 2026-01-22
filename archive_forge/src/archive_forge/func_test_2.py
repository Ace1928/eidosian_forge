from os.path import join, sep, dirname
from numpy.distutils.misc_util import (
from numpy.testing import (
def test_2(self):
    assert_equal(appendpath('prefix/sub', 'name'), join('prefix', 'sub', 'name'))
    assert_equal(appendpath('prefix/sub', 'sup/name'), join('prefix', 'sub', 'sup', 'name'))
    assert_equal(appendpath('/prefix/sub', '/prefix/name'), ajoin('prefix', 'sub', 'name'))