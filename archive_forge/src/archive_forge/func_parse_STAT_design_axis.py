from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_STAT_design_axis(self):
    assert self.is_cur_keyword_('DesignAxis')
    names = []
    axisTag = self.expect_tag_()
    if axisTag not in ('ital', 'opsz', 'slnt', 'wdth', 'wght') and (not axisTag.isupper()):
        log.warning(f'Unregistered axis tag {axisTag} should be uppercase.')
    axisOrder = self.expect_number_()
    self.expect_symbol_('{')
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.COMMENT:
            continue
        elif self.is_cur_keyword_('name'):
            location = self.cur_token_location_
            platformID, platEncID, langID, string = self.parse_stat_name_()
            name = self.ast.STATNameStatement('stat', platformID, platEncID, langID, string, location=location)
            names.append(name)
        elif self.cur_token_ == ';':
            continue
        else:
            raise FeatureLibError(f'Expected "name", got {self.cur_token_}', self.cur_token_location_)
    self.expect_symbol_('}')
    return self.ast.STATDesignAxisStatement(axisTag, axisOrder, names, self.cur_token_location_)