import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def createSpacer(self, elem):
    width = elem.findtext('property/size/width')
    height = elem.findtext('property/size/height')
    if width is None or height is None:
        size_args = ()
    else:
        size_args = (int(width), int(height))
    sizeType = self.wprops.getProperty(elem, 'sizeType', QtWidgets.QSizePolicy.Expanding)
    policy = (QtWidgets.QSizePolicy.Minimum, sizeType)
    if self.wprops.getProperty(elem, 'orientation') == QtCore.Qt.Horizontal:
        policy = (policy[1], policy[0])
    spacer = self.factory.createQObject('QSpacerItem', self.uniqueName('spacerItem'), size_args + policy, is_attribute=False)
    if self.stack.topIsLayout():
        lay = self.stack.peek()
        lp = elem.attrib['layout-position']
        if isinstance(lay, QtWidgets.QFormLayout):
            lay.setItem(lp[0], self._form_layout_role(lp), spacer)
        else:
            lay.addItem(spacer, *lp)