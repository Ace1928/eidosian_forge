import sys
from unittest import TestCase, main, skipUnless
from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by
def testInitDoesntWrapOnNonWindows(self):
    with osname('posix'):
        init()
        self.assertNotWrapped()