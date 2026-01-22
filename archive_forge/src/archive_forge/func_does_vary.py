from fontTools.varLib.models import VariationModel, normalizeValue, piecewiseLinearMap
@property
def does_vary(self):
    values = list(self.values.values())
    return any((v != values[0] for v in values[1:]))