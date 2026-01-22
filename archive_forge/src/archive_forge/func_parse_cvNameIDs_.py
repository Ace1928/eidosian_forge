from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_cvNameIDs_(self, tag, block_name):
    assert self.cur_token_ == block_name, self.cur_token_
    block = self.ast.NestedBlock(tag, block_name, location=self.cur_token_location_)
    self.expect_symbol_('{')
    for symtab in self.symbol_tables_:
        symtab.enter_scope()
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_(comments=True)
        if self.cur_token_type_ is Lexer.COMMENT:
            block.statements.append(self.ast.Comment(self.cur_token_, location=self.cur_token_location_))
        elif self.is_cur_keyword_('name'):
            location = self.cur_token_location_
            platformID, platEncID, langID, string = self.parse_name_()
            block.statements.append(self.ast.CVParametersNameStatement(tag, platformID, platEncID, langID, string, block_name, location=location))
        elif self.cur_token_ == ';':
            continue
        else:
            raise FeatureLibError('Expected "name"', self.cur_token_location_)
    self.expect_symbol_('}')
    for symtab in self.symbol_tables_:
        symtab.exit_scope()
    self.expect_symbol_(';')
    return block