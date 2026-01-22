from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagFromHead(self, token):
    self.parser.parseError('unexpected-start-tag-out-of-my-head', {'name': token['name']})
    self.tree.openElements.append(self.tree.headPointer)
    self.parser.phases['inHead'].processStartTag(token)
    for node in self.tree.openElements[::-1]:
        if node.name == 'head':
            self.tree.openElements.remove(node)
            break