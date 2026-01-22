import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_provided_values(self):
    av = dict(y=True, n=False, yes=True, no=False)
    self.assertIsTrue('y', av)
    self.assertIsTrue('Y', av)
    self.assertIsTrue('Yes', av)
    self.assertIsFalse('n', av)
    self.assertIsFalse('N', av)
    self.assertIsFalse('No', av)
    self.assertIsNone('1', av)
    self.assertIsNone('0', av)
    self.assertIsNone('on', av)
    self.assertIsNone('off', av)