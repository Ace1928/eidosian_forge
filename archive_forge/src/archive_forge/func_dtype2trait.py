import warnings
from .constants import ComparisonMode, DefaultValue
from .trait_base import SequenceTypes
from .trait_errors import TraitError
from .trait_type import TraitType
from .trait_types import Str, Any, Int as TInt, Float as TFloat
def dtype2trait(dtype):
    """ Get the corresponding trait for a numpy dtype.
    """
    import numpy
    if dtype.char in numpy.typecodes['Float']:
        return TFloat
    elif dtype.char in numpy.typecodes['AllInteger']:
        return TInt
    elif dtype.char[0] == 'S':
        return Str
    else:
        return Any