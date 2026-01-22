import pytest
import nibabel as nib
from nibabel.pkg_info import cmp_pkg_version
@pytest.mark.parametrize('args', [['foo.2'], ['foo.2', '1.0'], ['1.0', 'foo.2'], ['foo']])
def test_cmp_pkg_version_error(args):
    with pytest.raises(ValueError):
        cmp_pkg_version(*args)