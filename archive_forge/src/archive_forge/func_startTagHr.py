from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagHr(self, token):
    if self.tree.elementInScope('p', variant='button'):
        self.endTagP(impliedTagToken('p'))
    self.tree.insertElement(token)
    self.tree.openElements.pop()
    token['selfClosingAcknowledged'] = True
    self.parser.framesetOK = False