from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_anchor_marks_(self):
    anchorMarks = []
    while self.next_token_ == '<':
        anchor = self.parse_anchor_()
        if anchor is None and self.next_token_ != 'mark':
            continue
        self.expect_keyword_('mark')
        markClass = self.expect_markClass_reference_()
        anchorMarks.append((anchor, markClass))
    return anchorMarks