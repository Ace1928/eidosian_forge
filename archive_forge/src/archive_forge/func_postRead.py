import copy
from enum import IntEnum
from functools import reduce
from math import radians
import itertools
from collections import defaultdict, namedtuple
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.misc.arrayTools import quantizeRect
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform, Identity
from fontTools.misc.textTools import bytesjoin, pad, safeEval
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.transformPen import TransformPen
from .otBase import (
from fontTools.feaLib.lookupDebugInfo import LookupDebugInfo, LOOKUP_DEBUG_INFO_KEY
import logging
import struct
from typing import TYPE_CHECKING, Iterator, List, Optional, Set
def postRead(self, rawTable, font):
    clips = {}
    glyphOrder = font.getGlyphOrder()
    for i, rec in enumerate(rawTable['ClipRecord']):
        if rec.StartGlyphID > rec.EndGlyphID:
            log.warning('invalid ClipRecord[%i].StartGlyphID (%i) > EndGlyphID (%i); skipped', i, rec.StartGlyphID, rec.EndGlyphID)
            continue
        redefinedGlyphs = []
        missingGlyphs = []
        for glyphID in range(rec.StartGlyphID, rec.EndGlyphID + 1):
            try:
                glyph = glyphOrder[glyphID]
            except IndexError:
                missingGlyphs.append(glyphID)
                continue
            if glyph not in clips:
                clips[glyph] = copy.copy(rec.ClipBox)
            else:
                redefinedGlyphs.append(glyphID)
        if redefinedGlyphs:
            log.warning('ClipRecord[%i] overlaps previous records; ignoring redefined clip boxes for the following glyph ID range: [%i-%i]', i, min(redefinedGlyphs), max(redefinedGlyphs))
        if missingGlyphs:
            log.warning('ClipRecord[%i] range references missing glyph IDs: [%i-%i]', i, min(missingGlyphs), max(missingGlyphs))
    self.clips = clips