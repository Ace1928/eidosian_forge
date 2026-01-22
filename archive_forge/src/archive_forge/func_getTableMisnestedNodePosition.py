from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def getTableMisnestedNodePosition(self):
    """Get the foster parent element, and sibling to insert before
        (or None) when inserting a misnested table node"""
    lastTable = None
    fosterParent = None
    insertBefore = None
    for elm in self.openElements[::-1]:
        if elm.name == 'table':
            lastTable = elm
            break
    if lastTable:
        if lastTable.parent:
            fosterParent = lastTable.parent
            insertBefore = lastTable
        else:
            fosterParent = self.openElements[self.openElements.index(lastTable) - 1]
    else:
        fosterParent = self.openElements[0]
    return (fosterParent, insertBefore)