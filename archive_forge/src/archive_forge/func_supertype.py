from itertools import chain
from .coretypes import (Unit, int8, int16, int32, int64, uint8, uint16, uint32,
def supertype(measure):
    """Get the super type of a concrete numeric type

    Examples
    --------
    >>> supertype(int8)
    {signed}

    >>> supertype(float32)
    {floating}

    >>> supertype(complex128)
    {complexes}

    >>> supertype(bool_)
    {boolean}

    >>> supertype(Option(bool_))
    {boolean}
    """
    if isinstance(measure, Option):
        measure = measure.ty
    assert matches_typeset(measure, scalar), 'measure must be numeric'
    return supertype_map[measure]