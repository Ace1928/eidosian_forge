from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_feature_reference_(self):
    assert self.cur_token_ == 'feature', self.cur_token_
    location = self.cur_token_location_
    featureName = self.expect_tag_()
    self.expect_symbol_(';')
    return self.ast.FeatureReferenceStatement(featureName, location=location)