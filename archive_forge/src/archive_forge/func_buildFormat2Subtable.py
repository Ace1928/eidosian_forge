from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def buildFormat2Subtable(self, ruleset, classdefs, chaining=True):
    st = self.newSubtable_(chaining=chaining)
    st.Format = 2
    st.populateDefaults()
    if chaining:
        st.BacktrackClassDef, st.InputClassDef, st.LookAheadClassDef = [c.build() for c in classdefs]
    else:
        st.ClassDef = classdefs[1].build()
    inClasses = classdefs[1].classes()
    classSets = []
    for _ in inClasses:
        classSet = self.newRuleSet_(format=2, chaining=chaining)
        classSets.append(classSet)
    coverage = set()
    classRuleAttr = self.ruleAttr_(format=2, chaining=chaining)
    for rule in ruleset.rules:
        ruleAsSubtable = self.newRule_(format=2, chaining=chaining)
        if chaining:
            ruleAsSubtable.BacktrackGlyphCount = len(rule.prefix)
            ruleAsSubtable.LookAheadGlyphCount = len(rule.suffix)
            ruleAsSubtable.Backtrack = [st.BacktrackClassDef.classDefs[list(x)[0]] for x in reversed(rule.prefix)]
            ruleAsSubtable.LookAhead = [st.LookAheadClassDef.classDefs[list(x)[0]] for x in rule.suffix]
            ruleAsSubtable.InputGlyphCount = len(rule.glyphs)
            ruleAsSubtable.Input = [st.InputClassDef.classDefs[list(x)[0]] for x in rule.glyphs[1:]]
            setForThisRule = classSets[st.InputClassDef.classDefs[list(rule.glyphs[0])[0]]]
        else:
            ruleAsSubtable.GlyphCount = len(rule.glyphs)
            ruleAsSubtable.Class = [st.ClassDef.classDefs[list(x)[0]] for x in rule.glyphs[1:]]
            setForThisRule = classSets[st.ClassDef.classDefs[list(rule.glyphs[0])[0]]]
        self.buildLookupList(rule, ruleAsSubtable)
        coverage |= set(rule.glyphs[0])
        getattr(setForThisRule, classRuleAttr).append(ruleAsSubtable)
        setattr(setForThisRule, f'{classRuleAttr}Count', getattr(setForThisRule, f'{classRuleAttr}Count') + 1)
    setattr(st, self.ruleSetAttr_(format=2, chaining=chaining), classSets)
    setattr(st, self.ruleSetAttr_(format=2, chaining=chaining) + 'Count', len(classSets))
    st.Coverage = buildCoverage(coverage, self.glyphMap)
    return st