from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_ligatureCaretByIndex_(self):
    assert self.is_cur_keyword_('LigatureCaretByIndex')
    location = self.cur_token_location_
    glyphs = self.parse_glyphclass_(accept_glyphname=True)
    carets = [self.expect_number_()]
    while self.next_token_ != ';':
        carets.append(self.expect_number_())
    self.expect_symbol_(';')
    return self.ast.LigatureCaretByIndexStatement(glyphs, carets, location=location)