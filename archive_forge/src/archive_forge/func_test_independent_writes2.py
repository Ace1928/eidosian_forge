import threading
import pyomo.common.unittest as unittest
from pyomo.common.multithread import *
from threading import Thread
from pyomo.opt.base.solvers import check_available_solvers
def test_independent_writes2(self):
    sut = MultiThreadWrapper(Dummy)

    def thread_func():
        sut.number = 2
    t = Thread(target=thread_func)
    t.start()
    t.join()
    self.assertEqual(sut.number, 1)