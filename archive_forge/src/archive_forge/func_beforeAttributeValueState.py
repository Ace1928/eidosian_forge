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
def beforeAttributeValueState(self):
    data = self.stream.char()
    if data in spaceCharacters:
        self.stream.charsUntil(spaceCharacters, True)
    elif data == '"':
        self.state = self.attributeValueDoubleQuotedState
    elif data == '&':
        self.state = self.attributeValueUnQuotedState
        self.stream.unget(data)
    elif data == "'":
        self.state = self.attributeValueSingleQuotedState
    elif data == '>':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-attribute-value-but-got-right-bracket'})
        self.emitCurrentToken()
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.currentToken['data'][-1][1] += '�'
        self.state = self.attributeValueUnQuotedState
    elif data in ('=', '<', '`'):
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'equals-in-unquoted-attribute-value'})
        self.currentToken['data'][-1][1] += data
        self.state = self.attributeValueUnQuotedState
    elif data is EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-attribute-value-but-got-eof'})
        self.state = self.dataState
    else:
        self.currentToken['data'][-1][1] += data
        self.state = self.attributeValueUnQuotedState
    return True