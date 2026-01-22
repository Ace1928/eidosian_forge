import threading
import fasteners
from fasteners import test
def test_many_locked(self):
    obj = ManyLocks(10)
    obj.i_am_locked(lambda gotten: self.assertTrue(all(gotten)))
    obj.i_am_not_locked(lambda gotten: self.assertEqual(0, sum(gotten)))