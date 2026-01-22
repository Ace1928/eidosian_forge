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
def test_raw_construction_timer(self):
    a = ConstructionTimer(None)
    self.assertRegex(str(a), 'ConstructionTimer object for NoneType \\(unknown\\); [0-9\\.]+ elapsed seconds')
    v = Var()
    v.construct()
    a = ConstructionTimer(_VarData(v))
    self.assertRegex(str(a), 'ConstructionTimer object for Var ScalarVar\\[NOTSET\\]; [0-9\\.]+ elapsed seconds')