from fontTools.varLib.models import VariationModel, normalizeValue, piecewiseLinearMap
def get_deltas_and_supports(self, model_cache=None, avar=None):
    values = list(self.values.values())
    return self.model(model_cache, avar).getDeltasAndSupports(values)