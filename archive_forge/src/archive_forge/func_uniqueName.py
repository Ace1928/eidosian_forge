import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def uniqueName(self, name):
    """UIParser.uniqueName(string) -> string

        Create a unique name from a string.
        >>> p = UIParser(QtCore, QtGui, QtWidgets)
        >>> p.uniqueName("foo")
        'foo'
        >>> p.uniqueName("foo")
        'foo1'
        """
    try:
        suffix = self.name_suffixes[name]
    except KeyError:
        self.name_suffixes[name] = 0
        return name
    suffix += 1
    self.name_suffixes[name] = suffix
    return '%s%i' % (name, suffix)