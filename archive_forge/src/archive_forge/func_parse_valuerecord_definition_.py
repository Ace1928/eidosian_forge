from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_valuerecord_definition_(self, vertical):
    assert self.is_cur_keyword_('valueRecordDef')
    location = self.cur_token_location_
    value = self.parse_valuerecord_(vertical)
    name = self.expect_name_()
    self.expect_symbol_(';')
    vrd = self.ast.ValueRecordDefinition(name, value, location=location)
    self.valuerecords_.define(name, vrd)
    return vrd