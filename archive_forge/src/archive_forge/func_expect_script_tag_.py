from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_script_tag_(self):
    tag = self.expect_tag_()
    if tag == 'dflt':
        raise FeatureLibError('"dflt" is not a valid script tag; use "DFLT" instead', self.cur_token_location_)
    return tag