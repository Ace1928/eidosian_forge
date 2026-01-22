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
def characterErrorsUCS2(self, data):
    skip = False
    for match in invalid_unicode_re.finditer(data):
        if skip:
            continue
        codepoint = ord(match.group())
        pos = match.start()
        if _utils.isSurrogatePair(data[pos:pos + 2]):
            char_val = _utils.surrogatePairToCodepoint(data[pos:pos + 2])
            if char_val in non_bmp_invalid_codepoints:
                self.errors.append('invalid-codepoint')
            skip = True
        elif codepoint >= 55296 and codepoint <= 57343 and (pos == len(data) - 1):
            self.errors.append('invalid-codepoint')
        else:
            skip = False
            self.errors.append('invalid-codepoint')