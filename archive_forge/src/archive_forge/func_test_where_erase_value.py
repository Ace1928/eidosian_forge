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
@skip_win32
def test_where_erase_value():
    """Test that `where` does not access f_locals and erase values."""
    import pexpect
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'], env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.expect('IPython')
    child.expect('\n')
    child.expect_exact('In [1]')
    block = dedent('\n    def simple_f():\n         myvar = 1\n         print(myvar)\n         1/0\n         print(myvar)\n    simple_f()    ')
    for line in block.splitlines():
        child.sendline(line)
        child.expect_exact(line)
    child.expect_exact('ZeroDivisionError')
    child.expect_exact('In [2]:')
    child.sendline('%debug')
    child.expect('ipdb>')
    child.sendline('myvar')
    child.expect('1')
    child.expect('ipdb>')
    child.sendline('myvar = 2')
    child.expect_exact('ipdb>')
    child.sendline('myvar')
    child.expect_exact('2')
    child.expect('ipdb>')
    child.sendline('where')
    child.expect('ipdb>')
    child.sendline('myvar')
    child.expect_exact('2')
    child.expect('ipdb>')
    child.close()