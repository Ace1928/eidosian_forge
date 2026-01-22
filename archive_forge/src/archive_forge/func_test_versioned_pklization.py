import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_versioned_pklization(tmpdir):
    tmpdir.chdir()
    obj = Pickled()
    savepkl('./pickled.pkz', obj, versioning=True)
    with pytest.raises(Exception):
        with mock.patch('nipype.utils.tests.test_filemanip.Pickled', PickledBreaker), mock.patch('nipype.__version__', '0.0.0'):
            loadpkl('./pickled.pkz')