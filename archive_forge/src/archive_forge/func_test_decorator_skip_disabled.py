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
@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='issues on PyPy')
@skip_win32
def test_decorator_skip_disabled():
    """test that decorator frame skipping can be disabled"""
    child = _decorator_skip_setup()
    child.expect_exact('3     bar(3, 4)')
    for input_, expected in [('skip_predicates debuggerskip False', ''), ('skip_predicates', 'debuggerskip : False'), ('step', '---> 2     def wrapped_fn'), ('step', '----> 3         __debuggerskip__'), ('step', '----> 4         helper_1()'), ('step', '---> 1 def helper_1():'), ('next', '----> 2     helpers_helper()'), ('next', '--Return--'), ('next', '----> 5         __debuggerskip__ = False')]:
        child.expect('ipdb>')
        child.sendline(input_)
        child.expect_exact(input_)
        child.expect_exact(expected)
    child.close()