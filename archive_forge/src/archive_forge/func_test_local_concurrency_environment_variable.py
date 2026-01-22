import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_local_concurrency_environment_variable(self):
    self.overrideEnv('BRZ_CONCURRENCY', '2')
    self.assertEqual(2, osutils.local_concurrency(use_cache=False))
    self.overrideEnv('BRZ_CONCURRENCY', '3')
    self.assertEqual(3, osutils.local_concurrency(use_cache=False))
    self.overrideEnv('BRZ_CONCURRENCY', 'foo')
    self.assertEqual(1, osutils.local_concurrency(use_cache=False))