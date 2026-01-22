import logging
import os
import pyomo.environ as pyo
import pyomo.contrib.viewer.qt as myqt
from pyomo.common.fileutils import this_file_dir
def select_model(self):
    items = self.tableWidget.selectedItems()
    if len(items) == 0:
        return
    self.ui_data.model = self.models[items[0].row()]
    self.close()