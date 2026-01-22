from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def generateImpliedEndTags(self, exclude=None):
    name = self.openElements[-1].name
    if name in frozenset(('dd', 'dt', 'li', 'option', 'optgroup', 'p', 'rp', 'rt')) and name != exclude:
        self.openElements.pop()
        self.generateImpliedEndTags(exclude)