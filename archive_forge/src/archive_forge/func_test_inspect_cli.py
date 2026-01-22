import sys
import subprocess
import types as pytypes
import os.path
import numpy as np
import builtins
from numba.core import types
from numba.tests.support import TestCase, temp_directory
from numba.misc.help.inspector import inspect_function, inspect_module
def test_inspect_cli(self):
    cmdbase = [sys.executable, '-m', 'numba.misc.help.inspector']
    dirpath = temp_directory('{}.{}'.format(__name__, self.__class__.__name__))
    filename = os.path.join(dirpath, 'out')
    expected_file = filename + '.html'
    cmds = cmdbase + ['--file', filename, 'math']
    self.assertFalse(os.path.isfile(expected_file))
    subprocess.check_output(cmds)
    self.assertTrue(os.path.isfile(expected_file))
    cmds = cmdbase + ['--file', filename, '--format', 'rst', 'math']
    expected_file = filename + '.rst'
    self.assertFalse(os.path.isfile(expected_file))
    subprocess.check_output(cmds)
    self.assertTrue(os.path.isfile(expected_file))
    cmds = cmdbase + ['--file', filename, '--format', 'foo', 'math']
    with self.assertRaises(subprocess.CalledProcessError) as raises:
        subprocess.check_output(cmds, stderr=subprocess.STDOUT)
    self.assertIn("'foo' is not supported", raises.exception.stdout.decode())