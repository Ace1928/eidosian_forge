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
def parseGSUBGPOS(lines, font, tableTag):
    container = ttLib.getTableClass(tableTag)()
    lookupMap = DeferredMapping()
    featureMap = DeferredMapping()
    assert tableTag in ('GSUB', 'GPOS')
    log.debug('Parsing %s', tableTag)
    self = getattr(ot, tableTag)()
    self.Version = 65536
    fields = {'script table begin': ('ScriptList', lambda lines: parseScriptList(lines, featureMap)), 'feature table begin': ('FeatureList', lambda lines: parseFeatureList(lines, lookupMap, featureMap)), 'lookup': ('LookupList', None)}
    for attr, parser in fields.values():
        setattr(self, attr, None)
    while lines.peek() is not None:
        typ = lines.peek()[0].lower()
        if typ not in fields:
            log.debug('Skipping %s', lines.peek())
            next(lines)
            continue
        attr, parser = fields[typ]
        if typ == 'lookup':
            if self.LookupList is None:
                self.LookupList = ot.LookupList()
                self.LookupList.Lookup = []
            _, name, _ = lines.peek()
            lookup = parseLookup(lines, tableTag, font, lookupMap)
            if lookupMap is not None:
                assert name not in lookupMap, 'Duplicate lookup name: %s' % name
                lookupMap[name] = len(self.LookupList.Lookup)
            else:
                assert int(name) == len(self.LookupList.Lookup), '%d %d' % (name, len(self.Lookup))
            self.LookupList.Lookup.append(lookup)
        else:
            assert getattr(self, attr) is None, attr
            setattr(self, attr, parser(lines))
    if self.LookupList:
        self.LookupList.LookupCount = len(self.LookupList.Lookup)
    if lookupMap is not None:
        lookupMap.applyDeferredMappings()
        if os.environ.get(LOOKUP_DEBUG_ENV_VAR):
            if 'Debg' not in font:
                font['Debg'] = newTable('Debg')
                font['Debg'].data = {}
            debug = font['Debg'].data.setdefault(LOOKUP_DEBUG_INFO_KEY, {}).setdefault(tableTag, {})
            for name, lookup in lookupMap.items():
                debug[str(lookup)] = ['', name, '']
        featureMap.applyDeferredMappings()
    container.table = self
    return container