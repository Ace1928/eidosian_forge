from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
def buildVarData(varRegionIndices, items, optimize=True):
    self = ot.VarData()
    self.VarRegionIndex = list(varRegionIndices)
    regionCount = self.VarRegionCount = len(self.VarRegionIndex)
    records = self.Item = []
    if items:
        for item in items:
            assert len(item) == regionCount
            records.append(list(item))
    self.ItemCount = len(self.Item)
    self.calculateNumShorts(optimize=optimize)
    return self