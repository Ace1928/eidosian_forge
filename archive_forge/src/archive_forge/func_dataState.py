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
def dataState(self):
    data = self.stream.char()
    if data == '&':
        self.state = self.entityDataState
    elif data == '<':
        self.state = self.tagOpenState
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '\x00'})
    elif data is EOF:
        return False
    elif data in spaceCharacters:
        self.tokenQueue.append({'type': tokenTypes['SpaceCharacters'], 'data': data + self.stream.charsUntil(spaceCharacters, True)})
    else:
        chars = self.stream.charsUntil(('&', '<', '\x00'))
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': data + chars})
    return True