import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def createWidget(self, elem):
    self.column_counter = 0
    self.row_counter = 0
    self.item_nr = 0
    self.itemstack = []
    self.sorting_enabled = None
    widget_class = elem.attrib['class'].replace('::', '.')
    if widget_class == 'Line':
        widget_class = 'QFrame'
    parent = self.stack.topwidget
    if isinstance(parent, (QtWidgets.QDockWidget, QtWidgets.QMdiArea, QtWidgets.QScrollArea, QtWidgets.QStackedWidget, QtWidgets.QToolBox, QtWidgets.QTabWidget, QtWidgets.QWizard)):
        parent = None
    self.stack.push(self.setupObject(widget_class, parent, elem))
    if isinstance(self.stack.topwidget, QtWidgets.QTableWidget):
        if self.getProperty(elem, 'columnCount') is None:
            self.stack.topwidget.setColumnCount(len(elem.findall('column')))
        if self.getProperty(elem, 'rowCount') is None:
            self.stack.topwidget.setRowCount(len(elem.findall('row')))
    self.traverseWidgetTree(elem)
    widget = self.stack.popWidget()
    if isinstance(widget, QtWidgets.QTreeView):
        self.handleHeaderView(elem, 'header', widget.header())
    elif isinstance(widget, QtWidgets.QTableView):
        self.handleHeaderView(elem, 'horizontalHeader', widget.horizontalHeader())
        self.handleHeaderView(elem, 'verticalHeader', widget.verticalHeader())
    elif isinstance(widget, QtWidgets.QAbstractButton):
        bg_i18n = self.wprops.getAttribute(elem, 'buttonGroup')
        if bg_i18n is not None:
            try:
                bg_name = bg_i18n.string
            except AttributeError:
                bg_name = bg_i18n
            if not bg_name:
                bg_name = 'buttonGroup'
            try:
                bg = self.button_groups[bg_name]
            except KeyError:
                bg = self.button_groups[bg_name] = ButtonGroup()
            if bg.object is None:
                bg.object = self.factory.createQObject('QButtonGroup', bg_name, (self.toplevelWidget,))
                setattr(self.toplevelWidget, bg_name, bg.object)
                bg.object.setObjectName(bg_name)
                if not bg.exclusive:
                    bg.object.setExclusive(False)
            bg.object.addButton(widget)
    if self.sorting_enabled is not None:
        widget.setSortingEnabled(self.sorting_enabled)
        self.sorting_enabled = None
    if self.stack.topIsLayout():
        lay = self.stack.peek()
        lp = elem.attrib['layout-position']
        if isinstance(lay, QtWidgets.QFormLayout):
            lay.setWidget(lp[0], self._form_layout_role(lp), widget)
        else:
            lay.addWidget(widget, *lp)
    topwidget = self.stack.topwidget
    if isinstance(topwidget, QtWidgets.QToolBox):
        icon = self.wprops.getAttribute(elem, 'icon')
        if icon is not None:
            topwidget.addItem(widget, icon, self.wprops.getAttribute(elem, 'label'))
        else:
            topwidget.addItem(widget, self.wprops.getAttribute(elem, 'label'))
        tooltip = self.wprops.getAttribute(elem, 'toolTip')
        if tooltip is not None:
            topwidget.setItemToolTip(topwidget.indexOf(widget), tooltip)
    elif isinstance(topwidget, QtWidgets.QTabWidget):
        icon = self.wprops.getAttribute(elem, 'icon')
        if icon is not None:
            topwidget.addTab(widget, icon, self.wprops.getAttribute(elem, 'title'))
        else:
            topwidget.addTab(widget, self.wprops.getAttribute(elem, 'title'))
        tooltip = self.wprops.getAttribute(elem, 'toolTip')
        if tooltip is not None:
            topwidget.setTabToolTip(topwidget.indexOf(widget), tooltip)
    elif isinstance(topwidget, QtWidgets.QWizard):
        topwidget.addPage(widget)
    elif isinstance(topwidget, QtWidgets.QStackedWidget):
        topwidget.addWidget(widget)
    elif isinstance(topwidget, (QtWidgets.QDockWidget, QtWidgets.QScrollArea)):
        topwidget.setWidget(widget)
    elif isinstance(topwidget, QtWidgets.QMainWindow):
        if type(widget) == QtWidgets.QWidget:
            topwidget.setCentralWidget(widget)
        elif isinstance(widget, QtWidgets.QToolBar):
            tbArea = self.wprops.getAttribute(elem, 'toolBarArea')
            if tbArea is None:
                topwidget.addToolBar(widget)
            else:
                topwidget.addToolBar(tbArea, widget)
            tbBreak = self.wprops.getAttribute(elem, 'toolBarBreak')
            if tbBreak:
                topwidget.insertToolBarBreak(widget)
        elif isinstance(widget, QtWidgets.QMenuBar):
            topwidget.setMenuBar(widget)
        elif isinstance(widget, QtWidgets.QStatusBar):
            topwidget.setStatusBar(widget)
        elif isinstance(widget, QtWidgets.QDockWidget):
            dwArea = self.wprops.getAttribute(elem, 'dockWidgetArea')
            topwidget.addDockWidget(QtCore.Qt.DockWidgetArea(dwArea), widget)