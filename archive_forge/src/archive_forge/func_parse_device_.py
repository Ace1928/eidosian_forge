from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_device_(self):
    result = None
    self.expect_symbol_('<')
    self.expect_keyword_('device')
    if self.next_token_ == 'NULL':
        self.expect_keyword_('NULL')
    else:
        result = [(self.expect_number_(), self.expect_number_())]
        while self.next_token_ == ',':
            self.expect_symbol_(',')
            result.append((self.expect_number_(), self.expect_number_()))
        result = tuple(result)
    self.expect_symbol_('>')
    return result