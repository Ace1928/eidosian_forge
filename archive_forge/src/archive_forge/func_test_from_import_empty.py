import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
@unittest.expectedFailure
def test_from_import_empty(self):
    self.assertSetEqual(self.module_gatherer.complete(5, 'from '), {'zzabc', 'zzabd', 'zzefg'})
    self.assertSetEqual(self.module_gatherer.complete(6, 'from  '), {'zzabc', 'zzabd', 'zzefg'})