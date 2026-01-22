from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_glyph_pattern_(self, vertical):
    prefix, glyphs, lookups, values, suffix = ([], [], [], [], [])
    hasMarks = False
    while self.next_token_ not in {'by', 'from', ';', ','}:
        gc = self.parse_glyphclass_(accept_glyphname=True)
        marked = False
        if self.next_token_ == "'":
            self.expect_symbol_("'")
            hasMarks = marked = True
        if marked:
            if suffix:
                raise FeatureLibError("Unsupported contextual target sequence: at most one run of marked (') glyph/class names allowed", self.cur_token_location_)
            glyphs.append(gc)
        elif glyphs:
            suffix.append(gc)
        else:
            prefix.append(gc)
        if self.is_next_value_():
            values.append(self.parse_valuerecord_(vertical))
        else:
            values.append(None)
        lookuplist = None
        while self.next_token_ == 'lookup':
            if lookuplist is None:
                lookuplist = []
            self.expect_keyword_('lookup')
            if not marked:
                raise FeatureLibError('Lookups can only follow marked glyphs', self.cur_token_location_)
            lookup_name = self.expect_name_()
            lookup = self.lookups_.resolve(lookup_name)
            if lookup is None:
                raise FeatureLibError('Unknown lookup "%s"' % lookup_name, self.cur_token_location_)
            lookuplist.append(lookup)
        if marked:
            lookups.append(lookuplist)
    if not glyphs and (not suffix):
        assert lookups == []
        return ([], prefix, [None] * len(prefix), values, [], hasMarks)
    else:
        if any(values[:len(prefix)]):
            raise FeatureLibError('Positioning cannot be applied in the bactrack glyph sequence, before the marked glyph sequence.', self.cur_token_location_)
        marked_values = values[len(prefix):len(prefix) + len(glyphs)]
        if any(marked_values):
            if any(values[len(prefix) + len(glyphs):]):
                raise FeatureLibError('Positioning values are allowed only in the marked glyph sequence, or after the final glyph node when only one glyph node is marked.', self.cur_token_location_)
            values = marked_values
        elif values and values[-1]:
            if len(glyphs) > 1 or any(values[:-1]):
                raise FeatureLibError('Positioning values are allowed only in the marked glyph sequence, or after the final glyph node when only one glyph node is marked.', self.cur_token_location_)
            values = values[-1:]
        elif any(values):
            raise FeatureLibError('Positioning values are allowed only in the marked glyph sequence, or after the final glyph node when only one glyph node is marked.', self.cur_token_location_)
        return (prefix, glyphs, lookups, values, suffix, hasMarks)