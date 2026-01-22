import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
@pytest.mark.parametrize('filename, expected', [('foo.nii', ['foo.nii']), (['foo.nii'], ['foo.nii']), (('foo', 'bar'), ['foo', 'bar']), (12.34, None)])
def test_ensure_list(filename, expected):
    x = ensure_list(filename)
    assert x == expected