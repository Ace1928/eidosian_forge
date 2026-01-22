from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_STAT_axis_value_(self):
    assert self.is_cur_keyword_('AxisValue')
    self.expect_symbol_('{')
    locations = []
    names = []
    flags = 0
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_(comments=True)
        if self.cur_token_type_ is Lexer.COMMENT:
            continue
        elif self.is_cur_keyword_('name'):
            location = self.cur_token_location_
            platformID, platEncID, langID, string = self.parse_stat_name_()
            name = self.ast.STATNameStatement('stat', platformID, platEncID, langID, string, location=location)
            names.append(name)
        elif self.is_cur_keyword_('location'):
            location = self.parse_STAT_location()
            locations.append(location)
        elif self.is_cur_keyword_('flag'):
            flags = self.expect_stat_flags()
        elif self.cur_token_ == ';':
            continue
        else:
            raise FeatureLibError(f'Unexpected token {self.cur_token_} in AxisValue', self.cur_token_location_)
    self.expect_symbol_('}')
    if not names:
        raise FeatureLibError('Expected "Axis Name"', self.cur_token_location_)
    if not locations:
        raise FeatureLibError('Expected "Axis location"', self.cur_token_location_)
    if len(locations) > 1:
        for location in locations:
            if len(location.values) > 1:
                raise FeatureLibError(f'Only one value is allowed in a Format 4 Axis Value Record, but {len(location.values)} were found.', self.cur_token_location_)
        format4_tags = []
        for location in locations:
            tag = location.tag
            if tag in format4_tags:
                raise FeatureLibError(f'Axis tag {tag} already defined.', self.cur_token_location_)
            format4_tags.append(tag)
    return self.ast.STATAxisValueStatement(names, locations, flags, self.cur_token_location_)