import os
from pyomo.common.dependencies import attempt_import, UnavailableClass
from pyomo.scripting.pyomo_parser import add_subparser
import pyomo.contrib.viewer.qt as myqt
def kernel_pyomo_init(self, kc):
    kc.execute(self._kernel_cmd_import_qt_magic, silent=True)
    kc.execute(self._kernel_cmd_import_ui, silent=True)
    kc.execute(self._kernel_cmd_import_pyomo_env, silent=False)