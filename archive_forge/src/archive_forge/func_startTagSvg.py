from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagSvg(self, token):
    self.tree.reconstructActiveFormattingElements()
    self.parser.adjustSVGAttributes(token)
    self.parser.adjustForeignAttributes(token)
    token['namespace'] = namespaces['svg']
    self.tree.insertElement(token)
    if token['selfClosing']:
        self.tree.openElements.pop()
        token['selfClosingAcknowledged'] = True