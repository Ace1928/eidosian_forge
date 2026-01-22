import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_pklization(tmpdir):
    tmpdir.chdir()
    exc = Exception('There is something wrong here')
    savepkl('./except.pkz', exc)
    newexc = loadpkl('./except.pkz')
    assert exc.args == newexc.args
    assert os.getcwd() == tmpdir.strpath