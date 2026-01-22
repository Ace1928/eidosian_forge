from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagAppletMarqueeObject(self, token):
    if self.tree.elementInScope(token['name']):
        self.tree.generateImpliedEndTags()
    if self.tree.openElements[-1].name != token['name']:
        self.parser.parseError('end-tag-too-early', {'name': token['name']})
    if self.tree.elementInScope(token['name']):
        element = self.tree.openElements.pop()
        while element.name != token['name']:
            element = self.tree.openElements.pop()
        self.tree.clearActiveFormattingElements()