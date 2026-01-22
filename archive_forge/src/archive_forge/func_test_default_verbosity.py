import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
def test_default_verbosity(self):
    log = StringIO()
    with LoggingIntercept(log):
        self.handler = LogHandler(os.path.dirname(__file__), stream=self.stream)
    self.assertIn('LogHandler class has been deprecated', log.getvalue())
    logger.addHandler(self.handler)
    logger.setLevel(logging.WARNING)
    logger.warning('(warn)')
    lineno = getframeinfo(currentframe()).lineno - 1
    ans = 'WARNING: "[base]%stest_log.py", %d, test_default_verbosity\n    (warn)\n' % (os.path.sep, lineno)
    self.assertEqual(self.stream.getvalue(), ans)