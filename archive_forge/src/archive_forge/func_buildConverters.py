from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
def buildConverters(tableSpec, tableNamespace):
    """Given a table spec from otData.py, build a converter object for each
    field of the table. This is called for each table in otData.py, and
    the results are assigned to the corresponding class in otTables.py."""
    converters = []
    convertersByName = {}
    for tp, name, repeat, aux, descr in tableSpec:
        tableName = name
        if name.startswith('ValueFormat'):
            assert tp == 'uint16'
            converterClass = ValueFormat
        elif name.endswith('Count') or name in ('StructLength', 'MorphType'):
            converterClass = {'uint8': ComputedUInt8, 'uint16': ComputedUShort, 'uint32': ComputedULong}[tp]
        elif name == 'SubTable':
            converterClass = SubTable
        elif name == 'ExtSubTable':
            converterClass = ExtSubTable
        elif name == 'SubStruct':
            converterClass = SubStruct
        elif name == 'FeatureParams':
            converterClass = FeatureParams
        elif name in ('CIDGlyphMapping', 'GlyphCIDMapping'):
            converterClass = StructWithLength
        elif not tp in converterMapping and '(' not in tp:
            tableName = tp
            converterClass = Struct
        else:
            converterClass = eval(tp, tableNamespace, converterMapping)
        conv = converterClass(name, repeat, aux, description=descr)
        if conv.tableClass:
            tableClass = conv.tableClass
        elif tp in ('MortChain', 'MortSubtable', 'MorxChain'):
            tableClass = tableNamespace.get(tp)
        else:
            tableClass = tableNamespace.get(tableName)
        if not conv.tableClass:
            conv.tableClass = tableClass
        if name in ['SubTable', 'ExtSubTable', 'SubStruct']:
            conv.lookupTypes = tableNamespace['lookupTypes']
            for t in conv.lookupTypes.values():
                for cls in t.values():
                    convertersByName[cls.__name__] = Table(name, repeat, aux, cls)
        if name == 'FeatureParams':
            conv.featureParamTypes = tableNamespace['featureParamTypes']
            conv.defaultFeatureParams = tableNamespace['FeatureParams']
            for cls in conv.featureParamTypes.values():
                convertersByName[cls.__name__] = Table(name, repeat, aux, cls)
        converters.append(conv)
        assert name not in convertersByName, name
        convertersByName[name] = conv
    return (converters, convertersByName)