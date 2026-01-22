import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def setBuddies(self):
    for widget, buddy in self.wprops.buddies:
        DEBUG('%s is buddy of %s', buddy, widget.objectName())
        try:
            widget.setBuddy(getattr(self.toplevelWidget, buddy))
        except AttributeError:
            DEBUG('ERROR in ui spec: %s (buddy of %s) does not exist', buddy, widget.objectName())