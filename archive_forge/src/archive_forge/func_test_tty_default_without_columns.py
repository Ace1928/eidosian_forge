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
def test_tty_default_without_columns(self):
    self.overrideEnv('BRZ_COLUMNS', None)
    self.overrideEnv('COLUMNS', None)

    def terminal_size(w, h):
        return (42, 42)
    self.set_fake_tty()
    self.replace__terminal_size(terminal_size)
    self.assertEqual(42, osutils.terminal_width())