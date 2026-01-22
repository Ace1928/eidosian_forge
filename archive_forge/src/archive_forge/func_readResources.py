import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def readResources(self, elem):
    """
        Read a "resources" tag and add the module to import to the parser's
        list of them.
        """
    try:
        iterator = getattr(elem, 'iter')
    except AttributeError:
        iterator = getattr(elem, 'getiterator')
    for include in iterator('include'):
        loc = include.attrib.get('location')
        if loc and loc.endswith('.qrc'):
            mname = os.path.basename(loc[:-4] + self._resource_suffix)
            if mname not in self.resources:
                self.resources.append(mname)