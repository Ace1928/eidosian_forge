from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def group_to_finiteset(group):
    """
            Converts ``group`` to a :class:``FiniteSet`` if it is an
            iterable.
            """
    if iterable(group):
        return FiniteSet(*group)
    else:
        return group