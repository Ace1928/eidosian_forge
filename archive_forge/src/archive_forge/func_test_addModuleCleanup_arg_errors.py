import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_addModuleCleanup_arg_errors(self):
    cleanups = []

    def cleanup(*args, **kwargs):
        cleanups.append((args, kwargs))

    class Module(object):
        unittest.addModuleCleanup(cleanup, 1, 2, function='hello')
        with self.assertRaises(TypeError):
            unittest.addModuleCleanup(function=cleanup, arg='hello')
        with self.assertRaises(TypeError):
            unittest.addModuleCleanup()
    unittest.case.doModuleCleanups()
    self.assertEqual(cleanups, [((1, 2), {'function': 'hello'})])