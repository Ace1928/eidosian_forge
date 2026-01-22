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
def jumpTo(self, bytes):
    """Look for the next sequence of bytes matching a given sequence. If
        a match is found advance the position to the last byte of the match"""
    try:
        self._position = self.index(bytes, self.position) + len(bytes) - 1
    except ValueError:
        raise StopIteration
    return True