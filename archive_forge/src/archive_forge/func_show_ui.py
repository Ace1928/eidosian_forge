import os
from pyomo.common.dependencies import attempt_import, UnavailableClass
from pyomo.scripting.pyomo_parser import add_subparser
import pyomo.contrib.viewer.qt as myqt
def show_ui(self):
    kc = self.window.active_frontend.kernel_client
    kc.execute(self._kernel_cmd_show_ui.format(self.active_widget_name()), silent=True)