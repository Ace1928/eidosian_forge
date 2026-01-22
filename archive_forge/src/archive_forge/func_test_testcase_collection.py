import datetime
import multiprocessing
import os
import time
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Param
def test_testcase_collection(self):
    with TempfileManager.new_context() as TMP:
        tmpdir = TMP.create_tempdir()
        for fname in ('a.py', 'b.py', 'b.txt', 'c.py', 'c.sh', 'c.yml', 'd.sh', 'd.txt', 'e.sh'):
            with open(os.path.join(tmpdir, fname), 'w'):
                pass
        py_tests, sh_tests = unittest.BaselineTestDriver.gather_tests([tmpdir])
        self.assertEqual(py_tests, [(os.path.basename(tmpdir) + '_b', os.path.join(tmpdir, 'b.py'), os.path.join(tmpdir, 'b.txt'))])
        self.assertEqual(sh_tests, [(os.path.basename(tmpdir) + '_c', os.path.join(tmpdir, 'c.sh'), os.path.join(tmpdir, 'c.yml')), (os.path.basename(tmpdir) + '_d', os.path.join(tmpdir, 'd.sh'), os.path.join(tmpdir, 'd.txt'))])
        self.python_test_driver(*py_tests[0])
        _update_baselines = os.environ.pop('PYOMO_TEST_UPDATE_BASELINES', None)
        try:
            with open(os.path.join(tmpdir, 'b.py'), 'w') as FILE:
                FILE.write('print("Hello, World")\n')
            with self.assertRaises(self.failureException):
                self.python_test_driver(*py_tests[0])
            with open(os.path.join(tmpdir, 'b.txt'), 'r') as FILE:
                self.assertEqual(FILE.read(), '')
            os.environ['PYOMO_TEST_UPDATE_BASELINES'] = '1'
            with self.assertRaises(self.failureException):
                self.python_test_driver(*py_tests[0])
            with open(os.path.join(tmpdir, 'b.txt'), 'r') as FILE:
                self.assertEqual(FILE.read(), 'Hello, World\n')
        finally:
            os.environ.pop('PYOMO_TEST_UPDATE_BASELINES', None)
            if _update_baselines is not None:
                os.environ['PYOMO_TEST_UPDATE_BASELINES'] = _update_baselines
        self.shell_test_driver(*sh_tests[1])
        _update_baselines = os.environ.pop('PYOMO_TEST_UPDATE_BASELINES', None)
        try:
            with open(os.path.join(tmpdir, 'd.sh'), 'w') as FILE:
                FILE.write('echo "Hello, World"\n')
            with self.assertRaises(self.failureException):
                self.shell_test_driver(*sh_tests[1])
            with open(os.path.join(tmpdir, 'd.txt'), 'r') as FILE:
                self.assertEqual(FILE.read(), '')
            os.environ['PYOMO_TEST_UPDATE_BASELINES'] = '1'
            with self.assertRaises(self.failureException):
                self.shell_test_driver(*sh_tests[1])
            with open(os.path.join(tmpdir, 'd.txt'), 'r') as FILE:
                self.assertEqual(FILE.read(), 'Hello, World\n')
        finally:
            os.environ.pop('PYOMO_TEST_UPDATE_BASELINES', None)
            if _update_baselines is not None:
                os.environ['PYOMO_TEST_UPDATE_BASELINES'] = _update_baselines