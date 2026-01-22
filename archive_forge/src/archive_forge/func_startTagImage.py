from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagImage(self, token):
    self.parser.parseError('unexpected-start-tag-treated-as', {'originalName': 'image', 'newName': 'img'})
    self.processStartTag(impliedTagToken('img', 'StartTag', attributes=token['data'], selfClosing=token['selfClosing']))