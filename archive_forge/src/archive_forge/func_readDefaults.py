import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def readDefaults(self, elem):
    self.defaults['margin'] = int(elem.attrib['margin'])
    self.defaults['spacing'] = int(elem.attrib['spacing'])