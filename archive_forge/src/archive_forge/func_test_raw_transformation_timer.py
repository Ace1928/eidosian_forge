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
def test_raw_transformation_timer(self):
    a = TransformationTimer(None, 'fwd')
    self.assertIn('TransformationTimer object for NoneType (fwd); ', str(a))
    a = TransformationTimer(None)
    self.assertIn('TransformationTimer object for NoneType; ', str(a))