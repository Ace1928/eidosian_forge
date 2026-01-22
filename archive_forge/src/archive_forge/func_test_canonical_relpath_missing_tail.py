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
def test_canonical_relpath_missing_tail(self):
    os.mkdir('MixedCaseParent')
    actual = osutils.canonical_relpath(self.test_base_dir, 'mixedcaseparent/nochild')
    self.assertEqual('work/MixedCaseParent/nochild', actual)