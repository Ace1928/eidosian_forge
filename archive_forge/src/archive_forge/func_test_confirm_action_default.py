import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_confirm_action_default(self):
    base_ui = _mod_ui.NoninteractiveUIFactory()
    for answer in [True, False]:
        self.assertEqual(_mod_ui.ConfirmationUserInterfacePolicy(base_ui, answer, {}).confirm_action('Do something?', 'breezy.tests.do_something', {}), answer)