from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def parseRCDataRawtext(self, token, contentType):
    assert contentType in ('RAWTEXT', 'RCDATA')
    self.tree.insertElement(token)
    if contentType == 'RAWTEXT':
        self.tokenizer.state = self.tokenizer.rawtextState
    else:
        self.tokenizer.state = self.tokenizer.rcdataState
    self.originalPhase = self.phase
    self.phase = self.phases['text']