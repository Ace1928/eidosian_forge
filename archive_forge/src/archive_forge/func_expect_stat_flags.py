from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_stat_flags(self):
    value = 0
    flags = {'OlderSiblingFontAttribute': 1, 'ElidableAxisValueName': 2}
    while self.next_token_ != ';':
        if self.next_token_ in flags:
            name = self.expect_name_()
            value = value | flags[name]
        else:
            raise FeatureLibError(f'Unexpected STAT flag {self.cur_token_}', self.cur_token_location_)
    return value