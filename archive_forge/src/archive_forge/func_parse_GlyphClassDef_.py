from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_GlyphClassDef_(self):
    assert self.is_cur_keyword_('GlyphClassDef')
    location = self.cur_token_location_
    if self.next_token_ != ',':
        baseGlyphs = self.parse_glyphclass_(accept_glyphname=False)
    else:
        baseGlyphs = None
    self.expect_symbol_(',')
    if self.next_token_ != ',':
        ligatureGlyphs = self.parse_glyphclass_(accept_glyphname=False)
    else:
        ligatureGlyphs = None
    self.expect_symbol_(',')
    if self.next_token_ != ',':
        markGlyphs = self.parse_glyphclass_(accept_glyphname=False)
    else:
        markGlyphs = None
    self.expect_symbol_(',')
    if self.next_token_ != ';':
        componentGlyphs = self.parse_glyphclass_(accept_glyphname=False)
    else:
        componentGlyphs = None
    self.expect_symbol_(';')
    return self.ast.GlyphClassDefStatement(baseGlyphs, markGlyphs, ligatureGlyphs, componentGlyphs, location=location)