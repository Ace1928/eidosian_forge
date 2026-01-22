from __future__ import absolute_import, division, unicode_literals
from six import unichr as chr
from collections import deque, OrderedDict
from sys import version_info
from .constants import spaceCharacters
from .constants import entities
from .constants import asciiLetters, asciiUpper2Lower
from .constants import digits, hexDigits, EOF
from .constants import tokenTypes, tagTokenTypes
from .constants import replacementCharacters
from ._inputstream import HTMLInputStream
from ._trie import Trie
def selfClosingStartTagState(self):
    data = self.stream.char()
    if data == '>':
        self.currentToken['selfClosing'] = True
        self.emitCurrentToken()
    elif data is EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'unexpected-EOF-after-solidus-in-tag'})
        self.stream.unget(data)
        self.state = self.dataState
    else:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'unexpected-character-after-solidus-in-tag'})
        self.stream.unget(data)
        self.state = self.beforeAttributeNameState
    return True