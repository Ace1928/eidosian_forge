import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
@pytest.mark.parametrize('list, expected', [(['foo.nii'], 'foo.nii'), (['foo', 'bar'], ['foo', 'bar'])])
def test_simplify_list(list, expected):
    x = simplify_list(list)
    assert x == expected