import builtins
import os
import sys
import platform
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch
from IPython.core import debugger
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
from IPython.testing.decorators import skip_win32
import pytest
@pytest.mark.skip(reason='recently fail for unknown reason on CI')
@skip_win32
def test_decorator_skip():
    """test that decorator frames can be skipped."""
    child = _decorator_skip_setup()
    child.expect_exact('ipython-input-8')
    child.expect_exact('3     bar(3, 4)')
    child.expect('ipdb>')
    child.expect('ipdb>')
    child.sendline('step')
    child.expect_exact('step')
    child.expect_exact('--Call--')
    child.expect_exact('ipython-input-6')
    child.expect_exact('1 @pdb_skipped_decorator')
    child.sendline('s')
    child.expect_exact('return x * y')
    child.close()