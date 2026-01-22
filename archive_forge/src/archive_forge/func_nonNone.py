from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def nonNone(lst):
    return [l for l in lst if l is not None]