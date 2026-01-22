import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
def test_import_empty(self):
    self.assertSetEqual(self.module_gatherer.complete(13, 'import zzabc.'), {'zzabc.e', 'zzabc.f'})
    self.assertSetEqual(self.module_gatherer.complete(14, 'import  zzabc.'), {'zzabc.e', 'zzabc.f'})