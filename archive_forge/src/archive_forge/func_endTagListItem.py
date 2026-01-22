from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagListItem(self, token):
    if token['name'] == 'li':
        variant = 'list'
    else:
        variant = None
    if not self.tree.elementInScope(token['name'], variant=variant):
        self.parser.parseError('unexpected-end-tag', {'name': token['name']})
    else:
        self.tree.generateImpliedEndTags(exclude=token['name'])
        if self.tree.openElements[-1].name != token['name']:
            self.parser.parseError('end-tag-too-early', {'name': token['name']})
        node = self.tree.openElements.pop()
        while node.name != token['name']:
            node = self.tree.openElements.pop()