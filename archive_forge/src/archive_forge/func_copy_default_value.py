import warnings
from .constants import ComparisonMode, DefaultValue
from .trait_base import SequenceTypes
from .trait_errors import TraitError
from .trait_type import TraitType
from .trait_types import Str, Any, Int as TInt, Float as TFloat
def copy_default_value(self, value):
    """ Returns a copy of the default value (called from the C code on
            first reference to a trait with no current value).
        """
    return value.copy()