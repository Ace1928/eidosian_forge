from fontTools import ttLib
from fontTools.ttLib.tables._c_m_a_p import cmap_classes
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import ValueRecord, valueRecordFormatDict
from fontTools.otlLib import builder as otl
from contextlib import contextmanager
from fontTools.ttLib import newTable
from fontTools.feaLib.lookupDebugInfo import LOOKUP_DEBUG_ENV_VAR, LOOKUP_DEBUG_INFO_KEY
from operator import setitem
import os
import logging
def parsePair(lines, font, _lookupMap=None):
    self = ot.PairPos()
    self.ValueFormat1 = self.ValueFormat2 = 0
    typ = lines.peeks()[0].split()[0].lower()
    if typ in ('left', 'right'):
        self.Format = 1
        values = {}
        for line in lines:
            assert len(line) == 4, line
            side = line[0].split()[0].lower()
            assert side in ('left', 'right'), side
            what = line[0][len(side):].title().replace(' ', '')
            mask = valueRecordFormatDict[what][0]
            glyph1, glyph2 = makeGlyphs(line[1:3])
            value = int(line[3])
            if not glyph1 in values:
                values[glyph1] = {}
            if not glyph2 in values[glyph1]:
                values[glyph1][glyph2] = (ValueRecord(), ValueRecord())
            rec2 = values[glyph1][glyph2]
            if side == 'left':
                self.ValueFormat1 |= mask
                vr = rec2[0]
            else:
                self.ValueFormat2 |= mask
                vr = rec2[1]
            assert not hasattr(vr, what), (vr, what)
            setattr(vr, what, value)
        self.Coverage = makeCoverage(set(values.keys()), font)
        self.PairSet = []
        for glyph1 in self.Coverage.glyphs:
            values1 = values[glyph1]
            pairset = ot.PairSet()
            records = pairset.PairValueRecord = []
            for glyph2 in sorted(values1.keys(), key=font.getGlyphID):
                values2 = values1[glyph2]
                pair = ot.PairValueRecord()
                pair.SecondGlyph = glyph2
                pair.Value1 = values2[0]
                pair.Value2 = values2[1] if self.ValueFormat2 else None
                records.append(pair)
            pairset.PairValueCount = len(pairset.PairValueRecord)
            self.PairSet.append(pairset)
        self.PairSetCount = len(self.PairSet)
    elif typ.endswith('class'):
        self.Format = 2
        classDefs = [None, None]
        while lines.peeks()[0].endswith('class definition begin'):
            typ = lines.peek()[0][:-len('class definition begin')].lower()
            idx, klass = {'first': (0, ot.ClassDef1), 'second': (1, ot.ClassDef2)}[typ]
            assert classDefs[idx] is None
            classDefs[idx] = parseClassDef(lines, font, klass=klass)
        self.ClassDef1, self.ClassDef2 = classDefs
        self.Class1Count, self.Class2Count = (1 + max(c.classDefs.values()) for c in classDefs)
        self.Class1Record = [ot.Class1Record() for i in range(self.Class1Count)]
        for rec1 in self.Class1Record:
            rec1.Class2Record = [ot.Class2Record() for j in range(self.Class2Count)]
            for rec2 in rec1.Class2Record:
                rec2.Value1 = ValueRecord()
                rec2.Value2 = ValueRecord()
        for line in lines:
            assert len(line) == 4, line
            side = line[0].split()[0].lower()
            assert side in ('left', 'right'), side
            what = line[0][len(side):].title().replace(' ', '')
            mask = valueRecordFormatDict[what][0]
            class1, class2, value = (int(x) for x in line[1:4])
            rec2 = self.Class1Record[class1].Class2Record[class2]
            if side == 'left':
                self.ValueFormat1 |= mask
                vr = rec2.Value1
            else:
                self.ValueFormat2 |= mask
                vr = rec2.Value2
            assert not hasattr(vr, what), (vr, what)
            setattr(vr, what, value)
        for rec1 in self.Class1Record:
            for rec2 in rec1.Class2Record:
                rec2.Value1 = ValueRecord(self.ValueFormat1, rec2.Value1)
                rec2.Value2 = ValueRecord(self.ValueFormat2, rec2.Value2) if self.ValueFormat2 else None
        self.Coverage = makeCoverage(set(self.ClassDef1.classDefs.keys()), font)
    else:
        assert 0, typ
    return self