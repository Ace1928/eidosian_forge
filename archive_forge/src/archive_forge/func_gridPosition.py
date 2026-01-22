import sys
import logging
import os.path
import re
from pyside2uic.exceptions import NoSuchWidgetError
from pyside2uic.objcreator import QObjectCreator
from pyside2uic.properties import Properties
def gridPosition(elem):
    """gridPosition(elem) -> tuple

    Return the 4-tuple of (row, column, rowspan, colspan)
    for a widget element, or an empty tuple.
    """
    try:
        return (int(elem.attrib['row']), int(elem.attrib['column']), int(elem.attrib.get('rowspan', 1)), int(elem.attrib.get('colspan', 1)))
    except KeyError:
        return ()