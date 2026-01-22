import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_choose_prompt_explicit(self):
    with ui_testing.TextUIFactory('') as factory:
        factory.choose('prompt', '&yes\n&No\nmore &info')
        self.assertEqual('prompt ([y]es, [N]o, more [i]nfo): \n', factory.stderr.getvalue())