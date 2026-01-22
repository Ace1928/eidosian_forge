from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_tag_(self):
    self.advance_lexer_()
    if self.cur_token_type_ is not Lexer.NAME:
        raise FeatureLibError('Expected a tag', self.cur_token_location_)
    if len(self.cur_token_) > 4:
        raise FeatureLibError('Tags cannot be longer than 4 characters', self.cur_token_location_)
    return (self.cur_token_ + '    ')[:4]