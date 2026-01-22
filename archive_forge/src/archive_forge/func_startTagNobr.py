from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagNobr(self, token):
    self.tree.reconstructActiveFormattingElements()
    if self.tree.elementInScope('nobr'):
        self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'nobr', 'endName': 'nobr'})
        self.processEndTag(impliedTagToken('nobr'))
        self.tree.reconstructActiveFormattingElements()
    self.addFormattingElement(token)