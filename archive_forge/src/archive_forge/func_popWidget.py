import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def popWidget(self):
    widget = list.pop(self)
    DEBUG('pop widget %s %s' % (widget.metaObject().className(), widget.objectName()))
    for item in reversed(self):
        if isinstance(item, QtWidgets.QWidget):
            self.topwidget = item
            break
    else:
        self.topwidget = None
    DEBUG('new topwidget %s' % (self.topwidget,))
    return widget