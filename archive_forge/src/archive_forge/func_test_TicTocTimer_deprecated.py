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
def test_TicTocTimer_deprecated(self):
    timer = TicTocTimer()
    with LoggingIntercept() as LOG, capture_output() as out:
        timer.tic('msg', None, None)
    self.assertEqual(out.getvalue(), '')
    self.assertRegex(LOG.getvalue().replace('\n', ' ').strip(), "DEPRECATED: tic\\(\\): 'ostream' and 'logger' should be specified as keyword arguments( +\\([^\\)]+\\)){2}")
    with LoggingIntercept() as LOG, capture_output() as out:
        timer.toc('msg', True, None, None)
    self.assertEqual(out.getvalue(), '')
    self.assertRegex(LOG.getvalue().replace('\n', ' ').strip(), "DEPRECATED: toc\\(\\): 'delta', 'ostream', and 'logger' should be specified as keyword arguments( +\\([^\\)]+\\)){2}")
    timer = TicTocTimer()
    with LoggingIntercept() as LOG, capture_output() as out:
        timer.tic('msg %s, %s', None, None)
    self.assertIn('msg None, None', out.getvalue())
    self.assertEqual(LOG.getvalue(), '')
    with LoggingIntercept() as LOG, capture_output() as out:
        timer.toc('msg %s, %s, %s', True, None, None)
    self.assertIn('msg True, None, None', out.getvalue())
    self.assertEqual(LOG.getvalue(), '')