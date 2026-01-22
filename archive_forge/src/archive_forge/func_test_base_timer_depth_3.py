import pyomo.common.unittest as unittest
from pyomo.common.tee import capture_output
import gc
from io import StringIO
from itertools import zip_longest
import logging
import sys
import time
from pyomo.common.log import LoggingIntercept
from pyomo.common.timing import (
from pyomo.environ import (
from pyomo.core.base.var import _VarData
def test_base_timer_depth_3(self):
    timer = self.make_timer_depth_2_two_children()
    timer.flatten()
    self.assertAlmostEqual(timer.timers['root'].total_time, 0.12)
    self.assertAlmostEqual(timer.timers['a'].total_time, 0.7)
    self.assertAlmostEqual(timer.timers['b'].total_time, 1.86)
    self.assertAlmostEqual(timer.timers['c'].total_time, 2.27)
    self.assertAlmostEqual(timer.timers['d'].total_time, 0.05)