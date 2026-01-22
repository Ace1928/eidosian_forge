from fontTools.merge.unicode import is_Default_Ignorable
from fontTools.pens.recordingPen import DecomposingRecordingPen
import logging
def computeMegaCmap(merger, cmapTables):
    """Sets merger.cmap and merger.glyphOrder."""
    chosenCmapTables = []
    for fontIdx, table in enumerate(cmapTables):
        format4 = None
        format12 = None
        for subtable in table.tables:
            properties = (subtable.format, subtable.platformID, subtable.platEncID)
            if properties in _CmapUnicodePlatEncodings.BMP:
                format4 = subtable
            elif properties in _CmapUnicodePlatEncodings.FullRepertoire:
                format12 = subtable
            else:
                log.warning("Dropped cmap subtable from font '%s':\tformat %2s, platformID %2s, platEncID %2s", fontIdx, subtable.format, subtable.platformID, subtable.platEncID)
        if format12 is not None:
            chosenCmapTables.append((format12, fontIdx))
        elif format4 is not None:
            chosenCmapTables.append((format4, fontIdx))
    merger.cmap = cmap = {}
    fontIndexForGlyph = {}
    glyphSets = [None for f in merger.fonts] if hasattr(merger, 'fonts') else None
    for table, fontIdx in chosenCmapTables:
        for uni, gid in table.cmap.items():
            oldgid = cmap.get(uni, None)
            if oldgid is None:
                cmap[uni] = gid
                fontIndexForGlyph[gid] = fontIdx
            elif is_Default_Ignorable(uni) or uni in (9676,):
                continue
            elif oldgid != gid:
                if merger.duplicateGlyphsPerFont[fontIdx].get(oldgid) is None:
                    if glyphSets is not None:
                        oldFontIdx = fontIndexForGlyph[oldgid]
                        for idx in (fontIdx, oldFontIdx):
                            if glyphSets[idx] is None:
                                glyphSets[idx] = merger.fonts[idx].getGlyphSet()
                    merger.duplicateGlyphsPerFont[fontIdx][oldgid] = gid
                elif merger.duplicateGlyphsPerFont[fontIdx][oldgid] != gid:
                    log.warning("Dropped mapping from codepoint %#06X to glyphId '%s'", uni, gid)