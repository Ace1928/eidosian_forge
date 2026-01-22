from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_ignore_(self):
    assert self.is_cur_keyword_('ignore')
    location = self.cur_token_location_
    self.advance_lexer_()
    if self.cur_token_ in ['substitute', 'sub']:
        chainContext = self.parse_ignore_context_('sub')
        return self.ast.IgnoreSubstStatement(chainContext, location=location)
    if self.cur_token_ in ['position', 'pos']:
        chainContext = self.parse_ignore_context_('pos')
        return self.ast.IgnorePosStatement(chainContext, location=location)
    raise FeatureLibError('Expected "substitute" or "position"', self.cur_token_location_)