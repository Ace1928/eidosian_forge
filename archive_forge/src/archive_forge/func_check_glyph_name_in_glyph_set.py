from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def check_glyph_name_in_glyph_set(self, *names):
    """Adds a glyph name (just `start`) or glyph names of a
        range (`start` and `end`) which are not in the glyph set
        to the "missing list" for future error reporting.

        If no glyph set is present, does nothing.
        """
    if self.glyphNames_:
        for name in names:
            if name in self.glyphNames_:
                continue
            if name not in self.missing:
                self.missing[name] = self.cur_token_location_