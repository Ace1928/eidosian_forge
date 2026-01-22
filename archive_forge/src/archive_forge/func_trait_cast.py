from functools import partial
from .constants import DefaultValue
def trait_cast(obj):
    """ Convert to a CTrait if the object knows how, else return None.
    """
    try:
        return as_ctrait(obj)
    except TypeError:
        return None