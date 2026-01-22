import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
def test_from_attr_module(self):
    self.assertSetEqual(self.module_gatherer.complete(9, 'from os.p'), {'os.path'})