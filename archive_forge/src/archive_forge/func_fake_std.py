import sys
from unittest import TestCase, main, skipUnless
from ..ansitowin32 import StreamWrapper
from ..initialise import init, just_fix_windows_console, _wipe_internal_state_for_tests
from .utils import osname, replace_by
def fake_std():
    stdout = Mock()
    stdout.closed = False
    stdout.isatty.return_value = False
    stdout.fileno.return_value = 1
    sys.stdout = stdout
    stderr = Mock()
    stderr.closed = False
    stderr.isatty.return_value = True
    stderr.fileno.return_value = 2
    sys.stderr = stderr