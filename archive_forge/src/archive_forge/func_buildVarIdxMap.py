from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
def buildVarIdxMap(varIdxes, glyphOrder):
    self = ot.VarIdxMap()
    self.mapping = {g: v for g, v in zip(glyphOrder, varIdxes)}
    return self