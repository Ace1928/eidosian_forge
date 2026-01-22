from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def interpolateFromMasters(self, loc, masterValues, *, round=noRound):
    """Interpolate from master-values, at location loc."""
    scalars = self.getMasterScalars(loc)
    return self.interpolateFromValuesAndScalars(masterValues, scalars)