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
@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='issues on PyPy')
@skip_win32
def test_decorator_skip_with_breakpoint():
    """test that decorator frame skipping can be disabled"""
    import pexpect
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'
    env['PROMPT_TOOLKIT_NO_CPR'] = '1'
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'], env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.str_last_chars = 500
    child.expect('IPython')
    child.expect('\n')
    child.timeout = 5 * IPYTHON_TESTING_TIMEOUT_SCALE
    with NamedTemporaryFile(suffix='.py', dir='.', delete=True) as tf:
        name = tf.name[:-3].split('/')[-1]
        tf.write('\n'.join([dedent(x) for x in skip_decorators_blocks[:-1]]).encode())
        tf.flush()
        codeblock = f'from {name} import f'
        dedented_blocks = [codeblock, 'f()']
        in_prompt_number = 1
        for cblock in dedented_blocks:
            child.expect_exact(f'In [{in_prompt_number}]:')
            in_prompt_number += 1
            for line in cblock.splitlines():
                child.sendline(line)
                child.expect_exact(line)
            child.sendline('')
        child.expect_exact('47     bar(3, 4)')
        for input_, expected in [(f'b {name}.py:3', ''), ('step', '1---> 3     pass # should not stop here except'), ('step', '---> 38 @pdb_skipped_decorator'), ('continue', '')]:
            child.expect('ipdb>')
            child.sendline(input_)
            child.expect_exact(input_)
            child.expect_exact(expected)
    child.close()