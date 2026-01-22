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
def rcdataLessThanSignState(self):
    data = self.stream.char()
    if data == '/':
        self.temporaryBuffer = ''
        self.state = self.rcdataEndTagOpenState
    else:
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '<'})
        self.stream.unget(data)
        self.state = self.rcdataState
    return True