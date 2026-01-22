import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_run_send_fail(self):
    up_downs = []
    runner = runners.FiniteRunner(self.jumper)
    it = runner.run_iter('jump')
    up_downs.append(next(it))
    self.assertRaises(excp.NotFound, it.send, 'fail')
    it.close()
    self.assertEqual([('down', 'up')], up_downs)