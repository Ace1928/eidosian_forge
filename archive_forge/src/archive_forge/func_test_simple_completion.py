import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
def test_simple_completion(self):
    self.assertSetEqual(self.module_gatherer.complete(10, 'import zza'), {'zzabc', 'zzabd'})
    self.assertSetEqual(self.module_gatherer.complete(11, 'import  zza'), {'zzabc', 'zzabd'})