import sys
from PySide2 import QtCore, QtGui, QtWidgets
def updateChildItems(self, parent):
    dividerIndex = 0
    for group in self.settings.childGroups():
        childIndex = self.findChild(parent, group, dividerIndex)
        if childIndex != -1:
            child = self.childAt(parent, childIndex)
            child.setText(1, '')
            child.setText(2, '')
            child.setData(2, QtCore.Qt.UserRole, None)
            self.moveItemForward(parent, childIndex, dividerIndex)
        else:
            child = self.createItem(group, parent, dividerIndex)
        child.setIcon(0, self.groupIcon)
        dividerIndex += 1
        self.settings.beginGroup(group)
        self.updateChildItems(child)
        self.settings.endGroup()
    for key in self.settings.childKeys():
        childIndex = self.findChild(parent, key, 0)
        if childIndex == -1 or childIndex >= dividerIndex:
            if childIndex != -1:
                child = self.childAt(parent, childIndex)
                for i in range(child.childCount()):
                    self.deleteItem(child, i)
                self.moveItemForward(parent, childIndex, dividerIndex)
            else:
                child = self.createItem(key, parent, dividerIndex)
            child.setIcon(0, self.keyIcon)
            dividerIndex += 1
        else:
            child = self.childAt(parent, childIndex)
        value = self.settings.value(key)
        if value is None:
            child.setText(1, 'Invalid')
        else:
            child.setText(1, value.__class__.__name__)
        child.setText(2, VariantDelegate.displayText(value))
        child.setData(2, QtCore.Qt.UserRole, value)
    while dividerIndex < self.childCount(parent):
        self.deleteItem(parent, dividerIndex)