import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_getusername_unicode(self):
    ui = ui_testing.TextUIFactory('someuserሴ')
    username = ui.get_username('Hello %(host)s', host='someሴ')
    self.assertEqual('someuserሴ', username)
    self.assertEqual('Hello someሴ: ', ui.stderr.getvalue())
    self.assertEqual('', ui.stdout.getvalue())