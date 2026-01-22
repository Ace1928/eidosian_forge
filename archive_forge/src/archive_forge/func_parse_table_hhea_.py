from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_table_hhea_(self, table):
    statements = table.statements
    fields = ('CaretOffset', 'Ascender', 'Descender', 'LineGap')
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_(comments=True)
        if self.cur_token_type_ is Lexer.COMMENT:
            statements.append(self.ast.Comment(self.cur_token_, location=self.cur_token_location_))
        elif self.cur_token_type_ is Lexer.NAME and self.cur_token_ in fields:
            key = self.cur_token_.lower()
            value = self.expect_number_()
            statements.append(self.ast.HheaField(key, value, location=self.cur_token_location_))
            if self.next_token_ != ';':
                raise FeatureLibError('Incomplete statement', self.next_token_location_)
        elif self.cur_token_ == ';':
            continue
        else:
            raise FeatureLibError('Expected CaretOffset, Ascender, Descender or LineGap', self.cur_token_location_)