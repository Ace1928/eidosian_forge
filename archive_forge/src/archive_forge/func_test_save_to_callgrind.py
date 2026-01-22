import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def test_save_to_callgrind(self):
    path1 = self._temppath('callgrind')
    self.stats.save(path1)
    with open(path1) as f:
        self.assertEqual(f.readline(), 'events: Ticks\n')
    path2 = osutils.pathjoin(self.test_dir, 'callgrind.out.foo')
    self.stats.save(path2)
    with open(path2) as f:
        self.assertEqual(f.readline(), 'events: Ticks\n')
    path3 = self._temppath('txt')
    self.stats.save(path3, format='callgrind')
    with open(path3) as f:
        self.assertEqual(f.readline(), 'events: Ticks\n')