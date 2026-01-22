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
def beforeDoctypeNameState(self):
    data = self.stream.char()
    if data in spaceCharacters:
        pass
    elif data == '>':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-doctype-name-but-got-right-bracket'})
        self.currentToken['correct'] = False
        self.tokenQueue.append(self.currentToken)
        self.state = self.dataState
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.currentToken['name'] = '�'
        self.state = self.doctypeNameState
    elif data is EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-doctype-name-but-got-eof'})
        self.currentToken['correct'] = False
        self.tokenQueue.append(self.currentToken)
        self.state = self.dataState
    else:
        self.currentToken['name'] = data
        self.state = self.doctypeNameState
    return True