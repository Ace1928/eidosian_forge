import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
def test_unallowed_options(self):
    with self.assertRaisesRegex(ValueError, "'fmt' is not a valid option"):
        LegacyPyomoFormatter(fmt='%(message)')
    with self.assertRaisesRegex(ValueError, "'style' is not a valid option"):
        LegacyPyomoFormatter(style='%')