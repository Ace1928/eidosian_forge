import sys
from unittest import TestCase, main, skipUnless
from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by
@patch('colorama.initialise.atexit.register')
def testAtexitRegisteredOnlyOnce(self, mockRegister):
    init()
    self.assertTrue(mockRegister.called)
    mockRegister.reset_mock()
    init()
    self.assertFalse(mockRegister.called)