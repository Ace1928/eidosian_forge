import gc
import weakref
import greenlet
from . import TestCase
def test_dead_weakref(self):

    def _dead_greenlet():
        g = greenlet.greenlet(lambda: None)
        g.switch()
        return g
    o = weakref.ref(_dead_greenlet())
    gc.collect()
    self.assertEqual(o(), None)