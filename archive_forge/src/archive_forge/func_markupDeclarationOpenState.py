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
def markupDeclarationOpenState(self):
    charStack = [self.stream.char()]
    if charStack[-1] == '-':
        charStack.append(self.stream.char())
        if charStack[-1] == '-':
            self.currentToken = {'type': tokenTypes['Comment'], 'data': ''}
            self.state = self.commentStartState
            return True
    elif charStack[-1] in ('d', 'D'):
        matched = True
        for expected in (('o', 'O'), ('c', 'C'), ('t', 'T'), ('y', 'Y'), ('p', 'P'), ('e', 'E')):
            charStack.append(self.stream.char())
            if charStack[-1] not in expected:
                matched = False
                break
        if matched:
            self.currentToken = {'type': tokenTypes['Doctype'], 'name': '', 'publicId': None, 'systemId': None, 'correct': True}
            self.state = self.doctypeState
            return True
    elif charStack[-1] == '[' and self.parser is not None and self.parser.tree.openElements and (self.parser.tree.openElements[-1].namespace != self.parser.tree.defaultNamespace):
        matched = True
        for expected in ['C', 'D', 'A', 'T', 'A', '[']:
            charStack.append(self.stream.char())
            if charStack[-1] != expected:
                matched = False
                break
        if matched:
            self.state = self.cdataSectionState
            return True
    self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-dashes-or-doctype'})
    while charStack:
        self.stream.unget(charStack.pop())
    self.state = self.bogusCommentState
    return True