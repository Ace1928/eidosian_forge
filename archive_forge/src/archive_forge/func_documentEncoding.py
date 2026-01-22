from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
@property
def documentEncoding(self):
    """Name of the character encoding that was used to decode the input stream, or
        :obj:`None` if that is not determined yet

        """
    if not hasattr(self, 'tokenizer'):
        return None
    return self.tokenizer.stream.charEncoding[0].name