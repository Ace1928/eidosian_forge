import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import skip
from ctypes import *
import numpy as np
import llvmlite.binding as ll
from numba.core import utils
from numba.tests.support import (TestCase, tag, import_dynamic, temp_directory,
import unittest
def check_setup_nested_py(self, setup_py_file):
    import numba
    numba_path = os.path.abspath(os.path.dirname(os.path.dirname(numba.__file__)))
    env = dict(os.environ)
    if env.get('PYTHONPATH', ''):
        env['PYTHONPATH'] = numba_path + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = numba_path

    def run_python(args):
        p = subprocess.Popen([sys.executable] + args, cwd=self.usecase_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        out, _ = p.communicate()
        rc = p.wait()
        if rc != 0:
            self.fail('python failed with the following output:\n%s' % out.decode('utf-8', 'ignore'))
    run_python([setup_py_file, 'build_ext', '--inplace'])
    code = 'if 1:\n            import nested.pycc_compiled_module as lib\n            assert lib.get_const() == 42\n            res = lib.ones(3)\n            assert list(res) == [1.0, 1.0, 1.0]\n            '
    run_python(['-c', code])