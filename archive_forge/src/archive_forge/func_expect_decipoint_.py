from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_decipoint_(self):
    if self.next_token_type_ == Lexer.FLOAT:
        return self.expect_float_()
    elif self.next_token_type_ is Lexer.NUMBER:
        return self.expect_number_() / 10
    else:
        raise FeatureLibError('Expected an integer or floating-point number', self.cur_token_location_)