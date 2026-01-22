import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def handleItem(self, elem):
    if self.stack.topIsLayout():
        elem[0].attrib['layout-position'] = _layout_position(elem)
        self.traverseWidgetTree(elem)
    else:
        w = self.stack.topwidget
        if isinstance(w, QtWidgets.QComboBox):
            text = self.wprops.getProperty(elem, 'text')
            icon = self.wprops.getProperty(elem, 'icon')
            if icon:
                w.addItem(icon, '')
            else:
                w.addItem('')
            w.setItemText(self.item_nr, text)
        elif isinstance(w, QtWidgets.QListWidget):
            self.disableSorting(w)
            item = self.createWidgetItem('QListWidgetItem', elem, w.item, self.item_nr)
            w.addItem(item)
        elif isinstance(w, QtWidgets.QTreeWidget):
            if self.itemstack:
                parent, _ = self.itemstack[-1]
                _, nr_in_root = self.itemstack[0]
            else:
                parent = w
                nr_in_root = self.item_nr
            item = self.factory.createQObject('QTreeWidgetItem', 'item_%d' % len(self.itemstack), (parent,), False)
            if self.item_nr == 0 and (not self.itemstack):
                self.sorting_enabled = self.factory.invoke('__sortingEnabled', w.isSortingEnabled)
                w.setSortingEnabled(False)
            self.itemstack.append((item, self.item_nr))
            self.item_nr = 0
            titm = w.topLevelItem(nr_in_root)
            for child, nr_in_parent in self.itemstack[1:]:
                titm = titm.child(nr_in_parent)
            column = -1
            for prop in elem.findall('property'):
                c_prop = self.wprops.convert(prop)
                c_prop_name = prop.attrib['name']
                if c_prop_name == 'text':
                    column += 1
                    if c_prop:
                        titm.setText(column, c_prop)
                elif c_prop_name == 'statusTip':
                    item.setStatusTip(column, c_prop)
                elif c_prop_name == 'toolTip':
                    item.setToolTip(column, c_prop)
                elif c_prop_name == 'whatsThis':
                    item.setWhatsThis(column, c_prop)
                elif c_prop_name == 'font':
                    item.setFont(column, c_prop)
                elif c_prop_name == 'icon':
                    item.setIcon(column, c_prop)
                elif c_prop_name == 'background':
                    item.setBackground(column, c_prop)
                elif c_prop_name == 'foreground':
                    item.setForeground(column, c_prop)
                elif c_prop_name == 'flags':
                    item.setFlags(c_prop)
                elif c_prop_name == 'checkState':
                    item.setCheckState(column, c_prop)
            self.traverseWidgetTree(elem)
            _, self.item_nr = self.itemstack.pop()
        elif isinstance(w, QtWidgets.QTableWidget):
            row = int(elem.attrib['row'])
            col = int(elem.attrib['column'])
            self.disableSorting(w)
            item = self.createWidgetItem('QTableWidgetItem', elem, w.item, row, col)
            w.setItem(row, col, item)
        self.item_nr += 1