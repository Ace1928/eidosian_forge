from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def lay_out_group(group, local_hints):
    """
            If ``group`` is a set of objects, uses a ``DiagramGrid``
            to lay it out and returns the grid.  Otherwise returns the
            object (i.e., ``group``).  If ``local_hints`` is not
            empty, it is supplied to ``DiagramGrid`` as the dictionary
            of hints.  Otherwise, the ``hints`` argument of
            ``_handle_groups`` is used.
            """
    if isinstance(group, FiniteSet):
        for obj in group:
            obj_groups[obj] = group
        if local_hints:
            groups_grids[group] = DiagramGrid(diagram.subdiagram_from_objects(group), **local_hints)
        else:
            groups_grids[group] = DiagramGrid(diagram.subdiagram_from_objects(group), **hints)
    else:
        obj_groups[group] = group