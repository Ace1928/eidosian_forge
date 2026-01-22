from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def count_morphisms_undirected(A, B):
    """
            Counts how many processed morphisms there are between the
            two supplied objects.
            """
    return len([m for m in morphisms_str_info if {m.domain, m.codomain} == {A, B}])