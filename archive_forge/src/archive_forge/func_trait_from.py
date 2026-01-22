from functools import partial
from .constants import DefaultValue
def trait_from(obj):
    """ Returns a trait derived from its input.
    """
    from .trait_types import Any
    from .traits import Trait
    if obj is None:
        return Any().as_ctrait()
    try:
        return as_ctrait(obj)
    except TypeError:
        return Trait(obj)