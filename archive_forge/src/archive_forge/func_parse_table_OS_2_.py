from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_table_OS_2_(self, table):
    statements = table.statements
    numbers = ('FSType', 'TypoAscender', 'TypoDescender', 'TypoLineGap', 'winAscent', 'winDescent', 'XHeight', 'CapHeight', 'WeightClass', 'WidthClass', 'LowerOpSize', 'UpperOpSize')
    ranges = ('UnicodeRange', 'CodePageRange')
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_(comments=True)
        if self.cur_token_type_ is Lexer.COMMENT:
            statements.append(self.ast.Comment(self.cur_token_, location=self.cur_token_location_))
        elif self.cur_token_type_ is Lexer.NAME:
            key = self.cur_token_.lower()
            value = None
            if self.cur_token_ in numbers:
                value = self.expect_number_()
            elif self.is_cur_keyword_('Panose'):
                value = []
                for i in range(10):
                    value.append(self.expect_number_())
            elif self.cur_token_ in ranges:
                value = []
                while self.next_token_ != ';':
                    value.append(self.expect_number_())
            elif self.is_cur_keyword_('Vendor'):
                value = self.expect_string_()
            statements.append(self.ast.OS2Field(key, value, location=self.cur_token_location_))
        elif self.cur_token_ == ';':
            continue