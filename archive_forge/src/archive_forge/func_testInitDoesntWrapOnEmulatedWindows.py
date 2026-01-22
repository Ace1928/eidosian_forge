import sys
from unittest import TestCase, main, skipUnless
from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by
@patch('colorama.initialise.reset_all')
@patch('colorama.ansitowin32.winapi_test', lambda *_: False)
def testInitDoesntWrapOnEmulatedWindows(self, _):
    with osname('nt'):
        init()
        self.assertNotWrapped()