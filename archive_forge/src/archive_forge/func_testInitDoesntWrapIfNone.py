import sys
from unittest import TestCase, main, skipUnless
from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by
def testInitDoesntWrapIfNone(self):
    with replace_by(None):
        init()
        self.assertIsNone(sys.stdout)
        self.assertIsNone(sys.stderr)