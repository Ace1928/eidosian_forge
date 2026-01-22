from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def is_cur_keyword_(self, k):
    if self.cur_token_type_ is Lexer.NAME:
        if isinstance(k, type('')):
            return self.cur_token_ == k
        else:
            return self.cur_token_ in k
    return False