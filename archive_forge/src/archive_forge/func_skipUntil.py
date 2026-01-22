from __future__ import absolute_import, division, unicode_literals
from six import text_type
from six.moves import http_client, urllib
import codecs
import re
from io import BytesIO, StringIO
from tensorboard._vendor import webencodings
from .constants import EOF, spaceCharacters, asciiLetters, asciiUppercase
from .constants import _ReparseException
from . import _utils
def skipUntil(self, chars):
    p = self.position
    while p < len(self):
        c = self[p:p + 1]
        if c in chars:
            self._position = p
            return c
        p += 1
    self._position = p
    return None