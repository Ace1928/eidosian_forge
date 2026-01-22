from PySide2.QtCore import QDate, QFile, Qt, QTextStream
from PySide2.QtGui import (QFont, QIcon, QKeySequence, QTextCharFormat,
from PySide2.QtPrintSupport import QPrintDialog, QPrinter
from PySide2.QtWidgets import (QAction, QApplication, QDialog, QDockWidget,
import dockwidgets_rc
def insertCustomer(self, customer):
    if not customer:
        return
    customerList = customer.split(', ')
    document = self.textEdit.document()
    cursor = document.find('NAME')
    if not cursor.isNull():
        cursor.beginEditBlock()
        cursor.insertText(customerList[0])
        oldcursor = cursor
        cursor = document.find('ADDRESS')
        if not cursor.isNull():
            for i in customerList[1:]:
                cursor.insertBlock()
                cursor.insertText(i)
            cursor.endEditBlock()
        else:
            oldcursor.endEditBlock()