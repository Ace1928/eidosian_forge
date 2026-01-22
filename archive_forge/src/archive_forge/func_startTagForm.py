from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagForm(self, token):
    self.parser.parseError('unexpected-form-in-table')
    if self.tree.formPointer is None:
        self.tree.insertElement(token)
        self.tree.formPointer = self.tree.openElements[-1]
        self.tree.openElements.pop()