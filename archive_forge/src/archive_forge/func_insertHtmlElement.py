from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def insertHtmlElement(self):
    self.tree.insertRoot(impliedTagToken('html', 'StartTag'))
    self.parser.phase = self.parser.phases['beforeHead']