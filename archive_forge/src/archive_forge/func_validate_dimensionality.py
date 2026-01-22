import copy
from functools import wraps
import numpy as np
from . import markup
from .dimensionality import Dimensionality, p_dict
from .registry import unit_registry
from .decorators import with_doc
def validate_dimensionality(value):
    if isinstance(value, str):
        try:
            return unit_registry[value].dimensionality
        except (KeyError, UnicodeDecodeError):
            return unit_registry[str(value)].dimensionality
    elif isinstance(value, Quantity):
        validate_unit_quantity(value)
        return value.dimensionality
    elif isinstance(value, Dimensionality):
        return value.copy()
    else:
        raise TypeError('units must be a quantity, string, or dimensionality, got %s' % type(value))