from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_anchordef_(self):
    assert self.is_cur_keyword_('anchorDef')
    location = self.cur_token_location_
    x, y = (self.expect_number_(), self.expect_number_())
    contourpoint = None
    if self.next_token_ == 'contourpoint':
        self.expect_keyword_('contourpoint')
        contourpoint = self.expect_number_()
    name = self.expect_name_()
    self.expect_symbol_(';')
    anchordef = self.ast.AnchorDefinition(name, x, y, contourpoint=contourpoint, location=location)
    self.anchors_.define(name, anchordef)
    return anchordef