from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagBody(self, token):
    if not self.tree.elementInScope('body'):
        self.parser.parseError()
        return
    elif self.tree.openElements[-1].name != 'body':
        for node in self.tree.openElements[2:]:
            if node.name not in frozenset(('dd', 'dt', 'li', 'optgroup', 'option', 'p', 'rp', 'rt', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'body', 'html')):
                self.parser.parseError('expected-one-end-tag-but-got-another', {'gotName': 'body', 'expectedName': node.name})
                break
    self.parser.phase = self.parser.phases['afterBody']