import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
def test_run_brz_subprocess_env_del(self):
    """run_brz_subprocess can remove environment variables too."""
    self.assertFalse('EXISTANT_ENV_VAR' in os.environ)

    def check_environment():
        self.assertFalse('EXISTANT_ENV_VAR' in os.environ)
    os.environ['EXISTANT_ENV_VAR'] = 'set variable'
    self.check_popen_state = check_environment
    self.assertRaises(_DontSpawnProcess, self.start_brz_subprocess, [], env_changes={'EXISTANT_ENV_VAR': None})
    self.assertEqual('set variable', os.environ['EXISTANT_ENV_VAR'])
    del os.environ['EXISTANT_ENV_VAR']