import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_redirect_no_synchronize_stdout(self):
    r, w = os.pipe()
    os.dup2(w, 1)
    sys.stdout = os.fdopen(1, 'w', closefd=False)
    rd = tee.redirect_fd(synchronize=False)
    self._generate_output(rd)
    with os.fdopen(r, 'r') as FILE:
        os.close(w)
        os.close(1)
        self.assertEqual(FILE.read(), 'to_stdout_1\nto_stdout_2\nto_fd1_2\n')