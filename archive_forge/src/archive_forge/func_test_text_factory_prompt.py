import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_factory_prompt(self):
    with ui_testing.TextUIFactory() as factory:
        factory.prompt('foo %2e')
        self.assertEqual('', factory.stdout.getvalue())
        self.assertEqual('foo %2e', factory.stderr.getvalue())