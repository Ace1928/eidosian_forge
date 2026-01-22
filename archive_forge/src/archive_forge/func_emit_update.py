import logging
from pyomo.common.collections import ComponentMap
from pyomo.contrib.viewer.qt import *
import pyomo.environ as pyo
def emit_update(self):
    if not self._begin_update:
        self.updated.emit()