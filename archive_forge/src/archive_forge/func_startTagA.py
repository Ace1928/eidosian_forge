from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagA(self, token):
    afeAElement = self.tree.elementInActiveFormattingElements('a')
    if afeAElement:
        self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'a', 'endName': 'a'})
        self.endTagFormatting(impliedTagToken('a'))
        if afeAElement in self.tree.openElements:
            self.tree.openElements.remove(afeAElement)
        if afeAElement in self.tree.activeFormattingElements:
            self.tree.activeFormattingElements.remove(afeAElement)
    self.tree.reconstructActiveFormattingElements()
    self.addFormattingElement(token)