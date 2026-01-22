import gc
import weakref
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_circular_greenlet(self):

    class circular_greenlet(greenlet.greenlet):
        self = None
    o = circular_greenlet()
    o.self = o
    o = weakref.ref(o)
    gc.collect()
    self.assertIsNone(o())
    self.assertFalse(gc.garbage, gc.garbage)