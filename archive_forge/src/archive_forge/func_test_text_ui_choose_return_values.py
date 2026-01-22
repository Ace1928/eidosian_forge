import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_text_ui_choose_return_values(self):

    def choose():
        return factory.choose('', '&Yes\n&No\nMaybe\nmore &info', 3)
    stdin_text = 'y\nn\n \n no \nb\na\nd \nyes with garbage\nY\nnot an answer\nno\ninfo\nmore info\nMaybe\nfoo\n'
    with ui_testing.TextUIFactory(stdin_text) as factory:
        self.assertEqual(0, choose())
        self.assertEqual(1, choose())
        self.assertEqual(3, choose())
        self.assertEqual(1, choose())
        self.assertEqual(0, choose())
        self.assertEqual(1, choose())
        self.assertEqual(3, choose())
        self.assertEqual(2, choose())
        self.assertEqual('foo\n', factory.stdin.read())
        self.assertEqual('', factory.stdin.readline())
        self.assertEqual(None, choose())