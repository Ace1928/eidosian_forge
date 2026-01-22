from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def impliedTagToken(name, type='EndTag', attributes=None, selfClosing=False):
    if attributes is None:
        attributes = {}
    return {'type': tokenTypes[type], 'name': name, 'data': attributes, 'selfClosing': selfClosing}