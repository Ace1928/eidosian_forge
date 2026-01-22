from __future__ import absolute_import, division, unicode_literals
from six import text_type
import re
from codecs import register_error, xmlcharrefreplace_errors
from .constants import voidElements, booleanAttributes, spaceCharacters
from .constants import rcdataElements, entities, xmlEntities
from . import treewalkers, _utils
from xml.sax.saxutils import escape
def encodeStrict(self, string):
    assert isinstance(string, text_type)
    if self.encoding:
        return string.encode(self.encoding, 'strict')
    else:
        return string