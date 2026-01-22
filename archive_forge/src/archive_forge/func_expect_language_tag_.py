from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_language_tag_(self):
    tag = self.expect_tag_()
    if tag == 'DFLT':
        raise FeatureLibError('"DFLT" is not a valid language tag; use "dflt" instead', self.cur_token_location_)
    return tag