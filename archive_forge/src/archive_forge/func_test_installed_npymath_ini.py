from os.path import join, sep, dirname
from numpy.distutils.misc_util import (
from numpy.testing import (
def test_installed_npymath_ini():
    info = get_info('npymath')
    assert isinstance(info, dict)
    assert 'define_macros' in info