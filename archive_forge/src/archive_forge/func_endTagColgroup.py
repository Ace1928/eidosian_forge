from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagColgroup(self, token):
    if self.ignoreEndTagColgroup():
        assert self.parser.innerHTML
        self.parser.parseError()
    else:
        self.tree.openElements.pop()
        self.parser.phase = self.parser.phases['inTable']