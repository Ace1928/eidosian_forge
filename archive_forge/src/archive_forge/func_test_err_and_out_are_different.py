import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_err_and_out_are_different(self):
    with tee.TeeStream() as t:
        out = t.STDOUT
        self.assertIs(out, t.STDOUT)
        err = t.STDERR
        self.assertIs(err, t.STDERR)
        self.assertIsNot(out, err)