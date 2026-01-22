import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def setTaborder(self, elem):
    lastwidget = None
    for widget_elem in elem:
        widget = getattr(self.toplevelWidget, widget_elem.text)
        if lastwidget is not None:
            self.toplevelWidget.setTabOrder(lastwidget, widget)
        lastwidget = widget