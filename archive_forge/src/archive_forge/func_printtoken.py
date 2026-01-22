import keyword
import tokenize
from html import escape
from typing import List
from . import reflect
def printtoken(self, type, token, sCoordinates, eCoordinates, line):
    if hasattr(tokenize, 'ENCODING') and type == tokenize.ENCODING:
        self.encoding = token
        return
    if not isinstance(token, bytes):
        token = token.encode(self.encoding)
    srow, scol = sCoordinates
    erow, ecol = eCoordinates
    if self.currentLine < srow:
        self.writer(b'\n' * (srow - self.currentLine))
        self.currentLine, self.currentCol = (srow, 0)
    self.writer(b' ' * (scol - self.currentCol))
    if self.lastIdentifier:
        type = 'identifier'
        self.parameters = 1
    elif type == tokenize.NAME:
        if keyword.iskeyword(token):
            type = 'keyword'
        elif self.parameters:
            type = 'parameter'
        else:
            type = 'variable'
    else:
        type = tokenize.tok_name.get(type)
        assert type is not None
        type = type.lower()
    self.writer(token, type)
    self.currentCol = ecol
    self.currentLine += token.count(b'\n')
    if self.currentLine != erow:
        self.currentCol = 0
    self.lastIdentifier = token in (b'def', b'class')
    if token == b':':
        self.parameters = 0