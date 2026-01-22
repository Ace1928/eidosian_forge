from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def buildFeatureVariationRecord(conditionTable, substitutionRecords):
    """Build a FeatureVariationRecord."""
    fvr = ot.FeatureVariationRecord()
    fvr.ConditionSet = ot.ConditionSet()
    fvr.ConditionSet.ConditionTable = conditionTable
    fvr.ConditionSet.ConditionCount = len(conditionTable)
    fvr.FeatureTableSubstitution = ot.FeatureTableSubstitution()
    fvr.FeatureTableSubstitution.Version = 65536
    fvr.FeatureTableSubstitution.SubstitutionRecord = substitutionRecords
    fvr.FeatureTableSubstitution.SubstitutionCount = len(substitutionRecords)
    return fvr