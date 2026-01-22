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
def test_is_inside_any(self):
    SRC_FOO_C = osutils.pathjoin('src', 'foo.c')
    for dirs, fn in [(['src', 'doc'], SRC_FOO_C), (['src'], SRC_FOO_C), (['src'], 'src')]:
        self.assertTrue(osutils.is_inside_any(dirs, fn))
    for dirs, fn in [(['src'], 'srccontrol'), (['src'], 'srccontrol/foo')]:
        self.assertFalse(osutils.is_inside_any(dirs, fn))