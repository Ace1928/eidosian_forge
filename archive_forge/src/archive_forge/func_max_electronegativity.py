from __future__ import annotations
from itertools import chain, combinations
from pymatgen.core import Element
from pymatgen.core.composition import Composition
def max_electronegativity(self):
    """
        Returns:
            The maximum pairwise electronegativity difference.
        """
    maximum = 0
    for e1, e2 in combinations(self.elements, 2):
        if abs(Element(e1).X - Element(e2).X) > maximum:
            maximum = abs(Element(e1).X - Element(e2).X)
    return maximum