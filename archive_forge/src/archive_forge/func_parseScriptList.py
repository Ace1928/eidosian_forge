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
def parseScriptList(lines, featureMap=None):
    self = ot.ScriptList()
    records = []
    with lines.between('script table'):
        for line in lines:
            while len(line) < 4:
                line.append('')
            scriptTag, langSysTag, defaultFeature, features = line
            log.debug('Adding script %s language-system %s', scriptTag, langSysTag)
            langSys = ot.LangSys()
            langSys.LookupOrder = None
            if defaultFeature:
                setReference(mapFeature, featureMap, defaultFeature, setattr, langSys, 'ReqFeatureIndex')
            else:
                langSys.ReqFeatureIndex = 65535
            syms = stripSplitComma(features)
            langSys.FeatureIndex = theList = [3] * len(syms)
            for i, sym in enumerate(syms):
                setReference(mapFeature, featureMap, sym, setitem, theList, i)
            langSys.FeatureCount = len(langSys.FeatureIndex)
            script = [s for s in records if s.ScriptTag == scriptTag]
            if script:
                script = script[0].Script
            else:
                scriptRec = ot.ScriptRecord()
                scriptRec.ScriptTag = scriptTag + ' ' * (4 - len(scriptTag))
                scriptRec.Script = ot.Script()
                records.append(scriptRec)
                script = scriptRec.Script
                script.DefaultLangSys = None
                script.LangSysRecord = []
                script.LangSysCount = 0
            if langSysTag == 'default':
                script.DefaultLangSys = langSys
            else:
                langSysRec = ot.LangSysRecord()
                langSysRec.LangSysTag = langSysTag + ' ' * (4 - len(langSysTag))
                langSysRec.LangSys = langSys
                script.LangSysRecord.append(langSysRec)
                script.LangSysCount = len(script.LangSysRecord)
    for script in records:
        script.Script.LangSysRecord = sorted(script.Script.LangSysRecord, key=lambda rec: rec.LangSysTag)
    self.ScriptRecord = sorted(records, key=lambda rec: rec.ScriptTag)
    self.ScriptCount = len(self.ScriptRecord)
    return self