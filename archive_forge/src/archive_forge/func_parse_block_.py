from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_block_(self, block, vertical, stylisticset=None, size_feature=False, cv_feature=None):
    self.expect_symbol_('{')
    for symtab in self.symbol_tables_:
        symtab.enter_scope()
    statements = block.statements
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_(comments=True)
        if self.cur_token_type_ is Lexer.COMMENT:
            statements.append(self.ast.Comment(self.cur_token_, location=self.cur_token_location_))
        elif self.cur_token_type_ is Lexer.GLYPHCLASS:
            statements.append(self.parse_glyphclass_definition_())
        elif self.is_cur_keyword_('anchorDef'):
            statements.append(self.parse_anchordef_())
        elif self.is_cur_keyword_({'enum', 'enumerate'}):
            statements.append(self.parse_enumerate_(vertical=vertical))
        elif self.is_cur_keyword_('feature'):
            statements.append(self.parse_feature_reference_())
        elif self.is_cur_keyword_('ignore'):
            statements.append(self.parse_ignore_())
        elif self.is_cur_keyword_('language'):
            statements.append(self.parse_language_())
        elif self.is_cur_keyword_('lookup'):
            statements.append(self.parse_lookup_(vertical))
        elif self.is_cur_keyword_('lookupflag'):
            statements.append(self.parse_lookupflag_())
        elif self.is_cur_keyword_('markClass'):
            statements.append(self.parse_markClass_())
        elif self.is_cur_keyword_({'pos', 'position'}):
            statements.append(self.parse_position_(enumerated=False, vertical=vertical))
        elif self.is_cur_keyword_('script'):
            statements.append(self.parse_script_())
        elif self.is_cur_keyword_({'sub', 'substitute', 'rsub', 'reversesub'}):
            statements.append(self.parse_substitute_())
        elif self.is_cur_keyword_('subtable'):
            statements.append(self.parse_subtable_())
        elif self.is_cur_keyword_('valueRecordDef'):
            statements.append(self.parse_valuerecord_definition_(vertical))
        elif stylisticset and self.is_cur_keyword_('featureNames'):
            statements.append(self.parse_featureNames_(stylisticset))
        elif cv_feature and self.is_cur_keyword_('cvParameters'):
            statements.append(self.parse_cvParameters_(cv_feature))
        elif size_feature and self.is_cur_keyword_('parameters'):
            statements.append(self.parse_size_parameters_())
        elif size_feature and self.is_cur_keyword_('sizemenuname'):
            statements.append(self.parse_size_menuname_())
        elif self.cur_token_type_ is Lexer.NAME and self.cur_token_ in self.extensions:
            statements.append(self.extensions[self.cur_token_](self))
        elif self.cur_token_ == ';':
            continue
        else:
            raise FeatureLibError('Expected glyph class definition or statement: got {} {}'.format(self.cur_token_type_, self.cur_token_), self.cur_token_location_)
    self.expect_symbol_('}')
    for symtab in self.symbol_tables_:
        symtab.exit_scope()
    name = self.expect_name_()
    if name != block.name.strip():
        raise FeatureLibError('Expected "%s"' % block.name.strip(), self.cur_token_location_)
    self.expect_symbol_(';')
    has_single = False
    has_multiple = False
    for s in statements:
        if isinstance(s, self.ast.SingleSubstStatement):
            has_single = not any([s.prefix, s.suffix, s.forceChain])
        elif isinstance(s, self.ast.MultipleSubstStatement):
            has_multiple = not any([s.prefix, s.suffix, s.forceChain])
    if has_single and has_multiple:
        statements = []
        for s in block.statements:
            if isinstance(s, self.ast.SingleSubstStatement):
                glyphs = s.glyphs[0].glyphSet()
                replacements = s.replacements[0].glyphSet()
                if len(replacements) == 1:
                    replacements *= len(glyphs)
                for i, glyph in enumerate(glyphs):
                    statements.append(self.ast.MultipleSubstStatement(s.prefix, glyph, s.suffix, [replacements[i]], s.forceChain, location=s.location))
            else:
                statements.append(s)
        block.statements = statements