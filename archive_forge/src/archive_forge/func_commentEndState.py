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
def commentEndState(self):
    data = self.stream.char()
    if data == '>':
        self.tokenQueue.append(self.currentToken)
        self.state = self.dataState
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.currentToken['data'] += '--�'
        self.state = self.commentState
    elif data == '!':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'unexpected-bang-after-double-dash-in-comment'})
        self.state = self.commentEndBangState
    elif data == '-':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'unexpected-dash-after-double-dash-in-comment'})
        self.currentToken['data'] += data
    elif data is EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'eof-in-comment-double-dash'})
        self.tokenQueue.append(self.currentToken)
        self.state = self.dataState
    else:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'unexpected-char-in-comment'})
        self.currentToken['data'] += '--' + data
        self.state = self.commentState
    return True