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
def make_timer_depth_2_with_name_collision(self):
    timer = HierarchicalTimer()
    timer.start('root')
    timer.start('a')
    timer.start('b')
    timer.stop('b')
    timer.start('c')
    timer.stop('c')
    timer.stop('a')
    timer.start('b')
    timer.stop('b')
    timer.stop('root')
    timer.timers['root'].total_time = 5.0
    timer.timers['root'].timers['a'].total_time = 4.0
    timer.timers['root'].timers['a'].timers['b'].total_time = 1.1
    timer.timers['root'].timers['a'].timers['c'].total_time = 2.2
    timer.timers['root'].timers['b'].total_time = 0.11
    return timer