import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_get_boolean(self):
    stdin_text = "y\nn\n \n y \n no \nyes with garbage\nY\nnot an answer\nno\nI'm sure!\nyes\nNO\nfoo\n"
    with ui_testing.TextUIFactory(stdin_text) as factory:
        self.assertEqual(True, factory.get_boolean(''))
        self.assertEqual(False, factory.get_boolean(''))
        self.assertEqual(True, factory.get_boolean(''))
        self.assertEqual(False, factory.get_boolean(''))
        self.assertEqual(True, factory.get_boolean(''))
        self.assertEqual(False, factory.get_boolean(''))
        self.assertEqual(True, factory.get_boolean(''))
        self.assertEqual(False, factory.get_boolean(''))
        self.assertEqual('foo\n', factory.stdin.read())
        self.assertEqual('', factory.stdin.readline())
        self.assertEqual(False, factory.get_boolean(''))