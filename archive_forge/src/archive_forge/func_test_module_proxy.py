import warnings
import pytest
from nibabel import pkg_info
from nibabel.deprecated import (
from nibabel.tests.test_deprecator import TestDeprecatorFunc as _TestDF
def test_module_proxy():
    mp = ModuleProxy('nibabel.deprecated')
    assert hasattr(mp, 'ModuleProxy')
    assert mp.ModuleProxy is ModuleProxy
    assert repr(mp) == '<module proxy for nibabel.deprecated>'