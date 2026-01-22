import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def test_save_to_txt(self):
    path = self._temppath('txt')
    self.stats.save(path)
    with open(path) as f:
        lines = f.readlines()
        self.assertEqual(lines[0], _TXT_HEADER)