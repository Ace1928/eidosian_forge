from __future__ import annotations
import os
from io import StringIO
from typing import TypeVar
from twisted.scripts import trial
from twisted.trial import runner
from twisted.trial.test import packages
def test_recurseImport(self) -> str:
    d = self.runTrial('package')
    self.assertIn(d, '[ERROR]')
    self.assertIn(d, 'test_bad_module')
    self.assertIn(d, 'test_import_module')
    self.assertNotIn(d, '<module ')
    self.assertNotIn(d, 'IOError')
    return d