from __future__ import annotations
import collections
import json
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.util.coord import pbc_diff
@property
def specie(self) -> Element | Species | DummySpecies:
    """The Species/Element at the site. Only works for ordered sites. Otherwise
        an AttributeError is raised. Use this property sparingly. Robust
        design should make use of the property species instead. Note that the
        singular of species is also species. So the choice of this variable
        name is governed by programmatic concerns as opposed to grammar.

        Raises:
            AttributeError if Site is not ordered.
        """
    if not self.is_ordered:
        raise AttributeError('specie property only works for ordered sites!')
    return next(iter(self.species))