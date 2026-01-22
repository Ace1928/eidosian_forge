import sys
from unittest import mock
import types
import warnings
import unittest
import os
import subprocess
import threading
from numba import config, njit
from numba.tests.support import TestCase
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
def test_entrypoint_extension_sequence(self):
    env_copy = os.environ.copy()
    env_copy['_EP_MAGIC_TOKEN'] = str(self._EP_MAGIC_TOKEN)
    themod = self.__module__
    thecls = type(self).__name__
    methname = 'test_entrypoint_handles_type_extensions'
    injected_method = '%s.%s.%s' % (themod, thecls, methname)
    cmdline = [sys.executable, '-m', 'numba.runtests', injected_method]
    out, err = self.run_cmd(cmdline, env_copy)
    _DEBUG = False
    if _DEBUG:
        print(out, err)