from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def unescape_string_(self, string, encoding):
    if encoding == 'utf_16_be':
        s = re.sub('\\\\[0-9a-fA-F]{4}', self.unescape_unichr_, string)
    else:
        unescape = lambda m: self.unescape_byte_(m, encoding)
        s = re.sub('\\\\[0-9a-fA-F]{2}', unescape, string)
    utf16 = tobytes(s, 'utf_16_be', 'surrogatepass')
    return tostr(utf16, 'utf_16_be')