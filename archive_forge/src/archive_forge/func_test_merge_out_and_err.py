import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
@unittest.skipIf(not tee._peek_available, 'Requires the _mergedReader, but _peek_available==False')
def test_merge_out_and_err(self):
    a = StringIO()
    b = StringIO()
    assert tee._poll_interval <= 0.1
    with tee.TeeStream(a, b) as t:
        t.STDOUT.write('Hello\nWorld')
        t.STDOUT.flush()
        time.sleep(tee._poll_interval * 100)
        t.STDERR.write('interrupting\ncow')
        t.STDERR.flush()
        start_time = time.time()
        while 'cow' not in a.getvalue() and time.time() - start_time < 1:
            time.sleep(tee._poll_interval)
    acceptable_results = {'Hello\ninterrupting\ncowWorld', 'interrupting\ncowHello\nWorld'}
    self.assertIn(a.getvalue(), acceptable_results)
    self.assertEqual(b.getvalue(), a.getvalue())