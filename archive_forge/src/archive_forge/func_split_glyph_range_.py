from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def split_glyph_range_(self, name, location):
    parts = name.split('-')
    solutions = []
    for i in range(len(parts)):
        start, limit = ('-'.join(parts[0:i]), '-'.join(parts[i:]))
        if start in self.glyphNames_ and limit in self.glyphNames_:
            solutions.append((start, limit))
    if len(solutions) == 1:
        start, limit = solutions[0]
        return (start, limit)
    elif len(solutions) == 0:
        raise FeatureLibError('"%s" is not a glyph in the font, and it can not be split into a range of known glyphs' % name, location)
    else:
        ranges = ' or '.join(['"%s - %s"' % (s, l) for s, l in solutions])
        raise FeatureLibError('Ambiguous glyph range "%s"; please use %s to clarify what you mean' % (name, ranges), location)