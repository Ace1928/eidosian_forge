from pyomo.common.gc_manager import PauseGC
import gc
import pyomo.common.unittest as unittest
def test_gc_nested(self):
    pgc = PauseGC()
    with pgc:
        self.assertFalse(gc.isenabled())
        with PauseGC():
            self.assertFalse(gc.isenabled())
        self.assertFalse(gc.isenabled())
    self.assertTrue(gc.isenabled())