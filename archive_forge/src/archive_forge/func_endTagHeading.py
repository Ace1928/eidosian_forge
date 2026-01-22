from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagHeading(self, token):
    for item in headingElements:
        if self.tree.elementInScope(item):
            self.tree.generateImpliedEndTags()
            break
    if self.tree.openElements[-1].name != token['name']:
        self.parser.parseError('end-tag-too-early', {'name': token['name']})
    for item in headingElements:
        if self.tree.elementInScope(item):
            item = self.tree.openElements.pop()
            while item.name not in headingElements:
                item = self.tree.openElements.pop()
            break