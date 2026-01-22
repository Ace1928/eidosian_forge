from ..Qt import QtCore, QtWidgets
def topLevelItems(self):
    return [self.topLevelItem(i) for i in range(self.topLevelItemCount())]