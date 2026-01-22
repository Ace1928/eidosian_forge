from fontTools.ttLib.tables import otTables as ot
from .table_builder import TableUnbuilder
def unbuildColrV1(layerList, baseGlyphList):
    layers = []
    if layerList:
        layers = layerList.Paint
    unbuilder = LayerListUnbuilder(layers)
    return {rec.BaseGlyph: unbuilder.unbuildPaint(rec.Paint) for rec in baseGlyphList.BaseGlyphPaintRecord}