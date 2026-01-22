import os
import pytest
from pkg_resources import resource_filename as pkgrf
from ....utils.filemanip import md5
from ... import base as nib
def test_bunch_repr():
    b = nib.Bunch(b=2, c=3, a=dict(n=1, m=2))
    assert repr(b) == "Bunch(a={'m': 2, 'n': 1}, b=2, c=3)"