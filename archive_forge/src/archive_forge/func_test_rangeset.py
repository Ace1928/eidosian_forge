import pyomo.common.unittest as unittest
import sys
from importlib import import_module
from io import StringIO
from pyomo.common.log import LoggingIntercept
def test_rangeset(self):
    log = StringIO()
    with LoggingIntercept(log):
        from pyomo.core.base.set import RangeSet
    self.assertEqual(log.getvalue(), '')
    log = StringIO()
    with LoggingIntercept(log, 'pyomo'):
        rs = force_load('pyomo.core.base.rangeset')
    self.assertIn('The pyomo.core.base.rangeset module is deprecated.', log.getvalue().strip().replace('\n', ' '))
    self.assertIs(RangeSet, rs.RangeSet)
    log = StringIO()
    with LoggingIntercept(log, 'pyomo'):
        rs = force_load('pyomo.core.base.rangeset')
    self.assertIn('The pyomo.core.base.rangeset module is deprecated.', log.getvalue().strip().replace('\n', ' '))
    self.assertIs(RangeSet, rs.RangeSet)