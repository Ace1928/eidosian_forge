import os
import pytest
from pkg_resources import resource_filename as pkgrf
from ....utils.filemanip import md5
from ... import base as nib
def test_bunch_attribute():
    b = nib.Bunch(a=1, b=[2, 3], c=None)
    assert b.a == 1
    assert b.b == [2, 3]
    assert b.c is None