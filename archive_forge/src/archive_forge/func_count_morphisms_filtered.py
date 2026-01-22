from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def count_morphisms_filtered(dom, cod, curving):
    """
            Counts the processed morphisms which go out of ``dom``
            into ``cod`` with curving ``curving``.
            """
    return len([m for m, m_str_info in morphisms_str_info.items() if (m.domain, m.codomain) == (dom, cod) and m_str_info.curving == curving])