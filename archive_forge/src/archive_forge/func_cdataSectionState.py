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
def cdataSectionState(self):
    data = []
    while True:
        data.append(self.stream.charsUntil(']'))
        data.append(self.stream.charsUntil('>'))
        char = self.stream.char()
        if char == EOF:
            break
        else:
            assert char == '>'
            if data[-1][-2:] == ']]':
                data[-1] = data[-1][:-2]
                break
            else:
                data.append(char)
    data = ''.join(data)
    nullCount = data.count('\x00')
    if nullCount > 0:
        for _ in range(nullCount):
            self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        data = data.replace('\x00', '�')
    if data:
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': data})
    self.state = self.dataState
    return True