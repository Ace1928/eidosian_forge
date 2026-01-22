from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def remapFeatures(table, featureRemap):
    """Go through the scripts list, and remap feature indices."""
    for scriptIndex, script in enumerate(table.ScriptList.ScriptRecord):
        defaultLangSys = script.Script.DefaultLangSys
        if defaultLangSys is not None:
            _remapLangSys(defaultLangSys, featureRemap)
        for langSysRecordIndex, langSysRec in enumerate(script.Script.LangSysRecord):
            langSys = langSysRec.LangSys
            _remapLangSys(langSys, featureRemap)
    if hasattr(table, 'FeatureVariations') and table.FeatureVariations is not None:
        for fvr in table.FeatureVariations.FeatureVariationRecord:
            for ftsr in fvr.FeatureTableSubstitution.SubstitutionRecord:
                ftsr.FeatureIndex = featureRemap[ftsr.FeatureIndex]