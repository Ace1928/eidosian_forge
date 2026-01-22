from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_base_script_list_(self, count):
    assert self.cur_token_ in ('HorizAxis.BaseScriptList', 'VertAxis.BaseScriptList'), self.cur_token_
    scripts = [self.parse_base_script_record_(count)]
    while self.next_token_ == ',':
        self.expect_symbol_(',')
        scripts.append(self.parse_base_script_record_(count))
    self.expect_symbol_(';')
    return scripts