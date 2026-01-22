import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
def test_numbered_level(self):
    testname = 'test_numbered_level'
    self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
    logger.setLevel(logging.WARNING)
    logger.log(45, '(hi)')
    ans = 'Level 45: (hi)\n'
    self.assertEqual(self.stream.getvalue(), ans)
    logger.log(45, '')
    ans += 'Level 45:\n'
    self.assertEqual(self.stream.getvalue(), ans)
    logger.setLevel(logging.DEBUG)
    logger.log(45, '(hi)')
    lineno = getframeinfo(currentframe()).lineno - 1
    ans += 'Level 45: "[base]%stest_log.py", %d, %s\n    (hi)\n' % (os.path.sep, lineno, testname)
    self.assertEqual(self.stream.getvalue(), ans)
    logger.log(45, '')
    lineno = getframeinfo(currentframe()).lineno - 1
    ans += 'Level 45: "[base]%stest_log.py", %d, %s\n\n' % (os.path.sep, lineno, testname)
    self.assertEqual(self.stream.getvalue(), ans)