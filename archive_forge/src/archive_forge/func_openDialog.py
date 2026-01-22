from PySide2 import QtCore, QtGui, QtWidgets, QtPrintSupport
def openDialog(self):
    dialog = DetailsDialog('Enter Customer Details', self)
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        self.createLetter(dialog.senderName(), dialog.senderAddress(), dialog.orderItems(), dialog.sendOffers())