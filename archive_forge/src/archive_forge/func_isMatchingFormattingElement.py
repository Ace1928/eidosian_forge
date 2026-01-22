from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def isMatchingFormattingElement(self, node1, node2):
    return node1.name == node2.name and node1.namespace == node2.namespace and (node1.attributes == node2.attributes)