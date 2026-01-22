import logging
import os
import pyomo.environ as pyo
import pyomo.contrib.viewer.qt as myqt
from pyomo.common.fileutils import this_file_dir
def update_models(self):
    import __main__
    s = __main__.__dict__
    keys = []
    for k in s:
        if isinstance(s[k], pyo.Block):
            keys.append(k)
    self.tableWidget.clearContents()
    self.tableWidget.setRowCount(len(keys))
    self.models = []
    for row, k in enumerate(sorted(keys)):
        item = myqt.QTableWidgetItem()
        item.setText(k)
        self.tableWidget.setItem(row, 0, item)
        item = myqt.QTableWidgetItem()
        try:
            item.setText(s[k].name)
        except:
            item.setText('None')
        self.tableWidget.setItem(row, 1, item)
        item = myqt.QTableWidgetItem()
        item.setText(str(type(s[k])))
        self.tableWidget.setItem(row, 2, item)
        self.models.append(s[k])