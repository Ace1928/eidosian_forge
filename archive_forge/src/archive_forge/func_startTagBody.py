from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagBody(self, token):
    self.parser.parseError('unexpected-start-tag', {'name': 'body'})
    if len(self.tree.openElements) == 1 or self.tree.openElements[1].name != 'body':
        assert self.parser.innerHTML
    else:
        self.parser.framesetOK = False
        for attr, value in token['data'].items():
            if attr not in self.tree.openElements[1].attributes:
                self.tree.openElements[1].attributes[attr] = value