import os
import pytest
from pkg_resources import resource_filename as pkgrf
from ....utils.filemanip import md5
from ... import base as nib
@pytest.mark.parametrize('args', [{}, {'a': 1, 'b': [2, 3]}])
def test_bunch(args):
    b = nib.Bunch(**args)
    assert b.__dict__ == args