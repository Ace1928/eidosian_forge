from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_lookup_(self, vertical):
    assert self.is_cur_keyword_('lookup')
    location, name = (self.cur_token_location_, self.expect_name_())
    if self.next_token_ == ';':
        lookup = self.lookups_.resolve(name)
        if lookup is None:
            raise FeatureLibError('Unknown lookup "%s"' % name, self.cur_token_location_)
        self.expect_symbol_(';')
        return self.ast.LookupReferenceStatement(lookup, location=location)
    use_extension = False
    if self.next_token_ == 'useExtension':
        self.expect_keyword_('useExtension')
        use_extension = True
    block = self.ast.LookupBlock(name, use_extension, location=location)
    self.parse_block_(block, vertical)
    self.lookups_.define(name, block)
    return block