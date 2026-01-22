import os
from pyomo.common.dependencies import attempt_import, UnavailableClass
from pyomo.scripting.pyomo_parser import add_subparser
import pyomo.contrib.viewer.qt as myqt
def new_frontend_master(self):
    widget = super().new_frontend_master()
    self.kernel_pyomo_init(widget.kernel_client)
    return widget