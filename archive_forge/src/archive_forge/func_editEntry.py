from PySide2.QtCore import (Qt, Signal, QRegExp, QModelIndex,
from PySide2.QtWidgets import (QWidget, QTabWidget, QMessageBox, QTableView,
from tablemodel import TableModel
from newaddresstab import NewAddressTab
from adddialogwidget import AddDialogWidget
def editEntry(self):
    """ Edit an entry in the addressbook. """
    tableView = self.currentWidget()
    proxyModel = tableView.model()
    selectionModel = tableView.selectionModel()
    indexes = selectionModel.selectedRows()
    for index in indexes:
        row = proxyModel.mapToSource(index).row()
        ix = self.tableModel.index(row, 0, QModelIndex())
        name = self.tableModel.data(ix, Qt.DisplayRole)
        ix = self.tableModel.index(row, 1, QModelIndex())
        address = self.tableModel.data(ix, Qt.DisplayRole)
    addDialog = AddDialogWidget()
    addDialog.setWindowTitle('Edit a Contact')
    addDialog.nameText.setReadOnly(True)
    addDialog.nameText.setText(name)
    addDialog.addressText.setText(address)
    if addDialog.exec_():
        newAddress = addDialog.address
        if newAddress != address:
            ix = self.tableModel.index(row, 1, QModelIndex())
            self.tableModel.setData(ix, newAddress, Qt.EditRole)