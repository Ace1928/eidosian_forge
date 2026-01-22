from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_glyphclass_(self, accept_glyphname, accept_null=False):
    if accept_glyphname and self.next_token_type_ in (Lexer.NAME, Lexer.CID):
        if accept_null and self.next_token_ == 'NULL':
            self.advance_lexer_()
            return self.ast.NullGlyph(location=self.cur_token_location_)
        glyph = self.expect_glyph_()
        self.check_glyph_name_in_glyph_set(glyph)
        return self.ast.GlyphName(glyph, location=self.cur_token_location_)
    if self.next_token_type_ is Lexer.GLYPHCLASS:
        self.advance_lexer_()
        gc = self.glyphclasses_.resolve(self.cur_token_)
        if gc is None:
            raise FeatureLibError('Unknown glyph class @%s' % self.cur_token_, self.cur_token_location_)
        if isinstance(gc, self.ast.MarkClass):
            return self.ast.MarkClassName(gc, location=self.cur_token_location_)
        else:
            return self.ast.GlyphClassName(gc, location=self.cur_token_location_)
    self.expect_symbol_('[')
    location = self.cur_token_location_
    glyphs = self.ast.GlyphClass(location=location)
    while self.next_token_ != ']':
        if self.next_token_type_ is Lexer.NAME:
            glyph = self.expect_glyph_()
            location = self.cur_token_location_
            if '-' in glyph and self.glyphNames_ and (glyph not in self.glyphNames_):
                start, limit = self.split_glyph_range_(glyph, location)
                self.check_glyph_name_in_glyph_set(start, limit)
                glyphs.add_range(start, limit, self.make_glyph_range_(location, start, limit))
            elif self.next_token_ == '-':
                start = glyph
                self.expect_symbol_('-')
                limit = self.expect_glyph_()
                self.check_glyph_name_in_glyph_set(start, limit)
                glyphs.add_range(start, limit, self.make_glyph_range_(location, start, limit))
            else:
                if '-' in glyph and (not self.glyphNames_):
                    log.warning(str(FeatureLibError(f'Ambiguous glyph name that looks like a range: {glyph!r}', location)))
                self.check_glyph_name_in_glyph_set(glyph)
                glyphs.append(glyph)
        elif self.next_token_type_ is Lexer.CID:
            glyph = self.expect_glyph_()
            if self.next_token_ == '-':
                range_location = self.cur_token_location_
                range_start = self.cur_token_
                self.expect_symbol_('-')
                range_end = self.expect_cid_()
                self.check_glyph_name_in_glyph_set(f'cid{range_start:05d}', f'cid{range_end:05d}')
                glyphs.add_cid_range(range_start, range_end, self.make_cid_range_(range_location, range_start, range_end))
            else:
                glyph_name = f'cid{self.cur_token_:05d}'
                self.check_glyph_name_in_glyph_set(glyph_name)
                glyphs.append(glyph_name)
        elif self.next_token_type_ is Lexer.GLYPHCLASS:
            self.advance_lexer_()
            gc = self.glyphclasses_.resolve(self.cur_token_)
            if gc is None:
                raise FeatureLibError('Unknown glyph class @%s' % self.cur_token_, self.cur_token_location_)
            if isinstance(gc, self.ast.MarkClass):
                gc = self.ast.MarkClassName(gc, location=self.cur_token_location_)
            else:
                gc = self.ast.GlyphClassName(gc, location=self.cur_token_location_)
            glyphs.add_class(gc)
        else:
            raise FeatureLibError(f'Expected glyph name, glyph range, or glyph class reference, found {self.next_token_!r}', self.next_token_location_)
    self.expect_symbol_(']')
    return glyphs