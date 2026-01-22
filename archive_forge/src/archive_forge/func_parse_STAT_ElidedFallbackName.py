from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_STAT_ElidedFallbackName(self):
    assert self.is_cur_keyword_('ElidedFallbackName')
    self.expect_symbol_('{')
    names = []
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_()
        if self.is_cur_keyword_('name'):
            platformID, platEncID, langID, string = self.parse_stat_name_()
            nameRecord = self.ast.STATNameStatement('stat', platformID, platEncID, langID, string, location=self.cur_token_location_)
            names.append(nameRecord)
        elif self.cur_token_ != ';':
            raise FeatureLibError(f'Unexpected token {self.cur_token_} in ElidedFallbackName', self.cur_token_location_)
    self.expect_symbol_('}')
    if not names:
        raise FeatureLibError('Expected "name"', self.cur_token_location_)
    return names