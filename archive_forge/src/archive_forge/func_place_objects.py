from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def place_objects(pt, placed_objects):
    """
            Does depth-first search in the underlying graph of the
            diagram and places the objects en route.
            """
    new_pt = (pt[0], pt[1] + 1)
    for adjacent_obj in adjlists[grid[pt]]:
        if adjacent_obj in placed_objects:
            continue
        DiagramGrid._put_object(new_pt, adjacent_obj, grid, [])
        placed_objects.add(adjacent_obj)
        placed_objects.update(place_objects(new_pt, placed_objects))
        new_pt = (new_pt[0] + 1, new_pt[1])
    return placed_objects