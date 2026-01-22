from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def clearStackToTableContext(self):
    while self.tree.openElements[-1].name not in ('table', 'html'):
        self.tree.openElements.pop()