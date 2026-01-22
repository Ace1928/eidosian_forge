from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
def structure_contains_atom_environment(self, atom_symbol, ce_symbol):
    """
        Checks whether the structure contains a given atom in a given environment.

        Args:
            atom_symbol: Symbol of the atom.
            ce_symbol: Symbol of the coordination environment.

        Returns:
            bool: True if the coordination environment is found for the given atom.
        """
    for isite, site in enumerate(self.structure):
        if Element(atom_symbol) in site.species.element_composition and self.site_contains_environment(isite, ce_symbol):
            return True
    return False