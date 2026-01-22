import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def popLayout(self):
    layout = list.pop(self)
    DEBUG('pop layout %s %s' % (layout.metaObject().className(), layout.objectName()))
    return layout