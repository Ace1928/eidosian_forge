from PySide2.QtCore import (Qt, Signal, QRegExp, QModelIndex,
from PySide2.QtWidgets import (QWidget, QTabWidget, QMessageBox, QTableView,
from tablemodel import TableModel
from newaddresstab import NewAddressTab
from adddialogwidget import AddDialogWidget
def readFromFile(self, filename):
    """ Read contacts in from a file. """
    try:
        f = open(filename, 'rb')
        addresses = pickle.load(f)
    except IOError:
        QMessageBox.information(self, 'Unable to open file: %s' % filename)
    finally:
        f.close()
    if len(addresses) == 0:
        QMessageBox.information(self, 'No contacts in file: %s' % filename)
    else:
        for address in addresses:
            self.addEntry(address['name'], address['address'])