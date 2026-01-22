import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_add_and_do_ModuleCleanup(self):
    module_cleanups = []

    def module_cleanup1(*args, **kwargs):
        module_cleanups.append((3, args, kwargs))

    def module_cleanup2(*args, **kwargs):
        module_cleanups.append((4, args, kwargs))

    class Module(object):
        unittest.addModuleCleanup(module_cleanup1, 1, 2, 3, four='hello', five='goodbye')
        unittest.addModuleCleanup(module_cleanup2)
    self.assertEqual(unittest.case._module_cleanups, [(module_cleanup1, (1, 2, 3), dict(four='hello', five='goodbye')), (module_cleanup2, (), {})])
    unittest.case.doModuleCleanups()
    self.assertEqual(module_cleanups, [(4, (), {}), (3, (1, 2, 3), dict(four='hello', five='goodbye'))])
    self.assertEqual(unittest.case._module_cleanups, [])