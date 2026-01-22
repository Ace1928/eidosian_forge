from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagMeta(self, token):
    self.tree.insertElement(token)
    self.tree.openElements.pop()
    token['selfClosingAcknowledged'] = True
    attributes = token['data']
    if self.parser.tokenizer.stream.charEncoding[1] == 'tentative':
        if 'charset' in attributes:
            self.parser.tokenizer.stream.changeEncoding(attributes['charset'])
        elif 'content' in attributes and 'http-equiv' in attributes and (attributes['http-equiv'].lower() == 'content-type'):
            data = _inputstream.EncodingBytes(attributes['content'].encode('utf-8'))
            parser = _inputstream.ContentAttrParser(data)
            codec = parser.parse()
            self.parser.tokenizer.stream.changeEncoding(codec)