from __future__ import absolute_import, division, unicode_literals
from six import text_type
import re
from codecs import register_error, xmlcharrefreplace_errors
from .constants import voidElements, booleanAttributes, spaceCharacters
from .constants import rcdataElements, entities, xmlEntities
from . import treewalkers, _utils
from xml.sax.saxutils import escape
def serializeError(self, data='XXX ERROR MESSAGE NEEDED'):
    self.errors.append(data)
    if self.strict:
        raise SerializeError