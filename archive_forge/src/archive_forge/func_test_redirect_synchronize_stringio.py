import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_redirect_synchronize_stringio(self):
    r, w = os.pipe()
    os.dup2(w, 1)
    try:
        sys.stdout, out = (StringIO(), sys.stdout)
        rd = tee.redirect_fd(synchronize=True)
        self._generate_output(rd)
    finally:
        sys.stdout, out = (out, sys.stdout)
    self.assertEqual(out.getvalue(), 'to_stdout_2\n')
    with os.fdopen(r, 'r') as FILE:
        os.close(w)
        os.close(1)
        self.assertEqual(FILE.read(), 'to_fd1_2\n')