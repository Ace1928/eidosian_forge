import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_capture_output_stack_error(self):
    OUT1 = StringIO()
    OUT2 = StringIO()
    old = (sys.stdout, sys.stderr)
    try:
        a = tee.capture_output(OUT1)
        a.setup()
        b = tee.capture_output(OUT2)
        b.setup()
        with self.assertRaisesRegex(RuntimeError, 'Captured output does not match sys.stdout'):
            a.reset()
        b.tee = None
    finally:
        sys.stdout, sys.stderr = old