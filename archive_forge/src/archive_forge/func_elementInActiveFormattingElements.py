from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def elementInActiveFormattingElements(self, name):
    """Check if an element exists between the end of the active
        formatting elements and the last marker. If it does, return it, else
        return false"""
    for item in self.activeFormattingElements[::-1]:
        if item == Marker:
            break
        elif item.name == name:
            return item
    return False