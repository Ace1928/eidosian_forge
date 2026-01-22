from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
def test_grouped_violinplot(self):
    graphics.grouped_violinplot(self.A, self.B)