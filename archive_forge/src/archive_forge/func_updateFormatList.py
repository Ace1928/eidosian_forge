from .. import exporters as exporters
from .. import functions as fn
from ..graphicsItems.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtWidgets
from . import exportDialogTemplate_generic as ui_template
def updateFormatList(self):
    current = self.ui.formatList.currentItem()
    self.ui.formatList.clear()
    gotCurrent = False
    for exp in exporters.listExporters():
        item = FormatExportListWidgetItem(exp, QtCore.QCoreApplication.translate('Exporter', exp.Name))
        self.ui.formatList.addItem(item)
        if item is current:
            self.ui.formatList.setCurrentRow(self.ui.formatList.count() - 1)
            gotCurrent = True
    if not gotCurrent:
        self.ui.formatList.setCurrentRow(0)