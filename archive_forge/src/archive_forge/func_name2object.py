import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def name2object(obj):
    if obj == self.uiname:
        return self.toplevelWidget
    else:
        return getattr(self.toplevelWidget, obj)