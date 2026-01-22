import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_non_terminal(self):
    """Even on non-ttys, make_ui_for_terminal gives a text ui."""
    stdin = stderr = stdout = ui_testing.StringIOWithEncoding()
    for term_type in ['dumb', None, 'xterm']:
        self.overrideEnv('TERM', term_type)
        uif = _mod_ui.make_ui_for_terminal(stdin, stdout, stderr)
        self.assertIsInstance(uif, _mod_ui_text.TextUIFactory, 'TERM={!r}'.format(term_type))