import threading
import pyomo.common.unittest as unittest
from pyomo.common.multithread import *
from threading import Thread
from pyomo.opt.base.solvers import check_available_solvers
def test_independent_writes(self):
    sut = MultiThreadWrapper(Dummy)
    sut.number = 2

    def thread_func():
        self.assertEqual(sut.number, 1)
    t = Thread(target=thread_func)
    t.start()
    t.join()