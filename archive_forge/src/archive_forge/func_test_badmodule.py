from __future__ import annotations
import os
from io import StringIO
from typing import TypeVar
from twisted.scripts import trial
from twisted.trial import runner
from twisted.trial.test import packages
def test_badmodule(self) -> str:
    d = self.runTrial('package.test_bad_module')
    self.assertIn(d, '[ERROR]')
    self.assertIn(d, 'package.test_bad_module')
    self.assertNotIn(d, 'IOError')
    self.assertNotIn(d, '<module ')
    return d