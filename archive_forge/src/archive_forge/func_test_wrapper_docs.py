import threading
import pyomo.common.unittest as unittest
from pyomo.common.multithread import *
from threading import Thread
from pyomo.opt.base.solvers import check_available_solvers
def test_wrapper_docs(self):
    sut = MultiThreadWrapper(Dummy)
    self.assertEqual(sut.__class__.__doc__, Dummy.__doc__)