import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def setArray(self, elem, name, setter):
    array = elem.attrib.get(name)
    if array:
        for idx, value in enumerate(array.split(',')):
            value = int(value)
            if value > 0:
                setter(idx, value)