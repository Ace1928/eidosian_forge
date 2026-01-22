from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def getSubModel(self, items):
    """Return a sub-model and the items that are not None.

        The sub-model is necessary for working with the subset
        of items when some are None.

        The sub-model is cached."""
    if None not in items:
        return (self, items)
    key = tuple((v is not None for v in items))
    subModel = self._subModels.get(key)
    if subModel is None:
        subModel = VariationModel(subList(key, self.origLocations), self.axisOrder)
        self._subModels[key] = subModel
    return (subModel, subList(key, items))