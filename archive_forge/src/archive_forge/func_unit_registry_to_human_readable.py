from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def unit_registry_to_human_readable(unit_registry):
    """Serialization of a unit registry."""
    if unit_registry is None:
        return None
    new_registry = {}
    integer_one = 1
    for k in SI_base_registry:
        if unit_registry[k] is integer_one:
            new_registry[k] = (1, 1)
        else:
            dim_list = list(unit_registry[k].dimensionality)
            if len(dim_list) != 1:
                raise TypeError('Compound units not allowed: {}'.format(dim_list))
            u_symbol = dim_list[0].u_symbol
            new_registry[k] = (float(unit_registry[k]), u_symbol)
    return new_registry