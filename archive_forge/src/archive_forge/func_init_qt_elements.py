import os
from pyomo.common.dependencies import attempt_import, UnavailableClass
from pyomo.scripting.pyomo_parser import add_subparser
import pyomo.contrib.viewer.qt as myqt
def init_qt_elements(self):
    super().init_qt_elements()
    self.kernel_pyomo_init(self.widget.kernel_client)
    self.run_script_act = myqt.QAction('&Run Script...', self.window)
    self.show_ui_act = myqt.QAction('&Show Pyomo Model Viewer', self.window)
    self.hide_ui_act = myqt.QAction('&Hide Pyomo Model Viewer', self.window)
    self.window.file_menu.addSeparator()
    self.window.file_menu.addAction(self.run_script_act)
    self.window.view_menu.addSeparator()
    self.window.view_menu.addAction(self.show_ui_act)
    self.window.view_menu.addAction(self.hide_ui_act)
    self.window.view_menu.addSeparator()
    self.run_script_act.triggered.connect(self.run_script)
    self.show_ui_act.triggered.connect(self.show_ui)
    self.hide_ui_act.triggered.connect(self.hide_ui)