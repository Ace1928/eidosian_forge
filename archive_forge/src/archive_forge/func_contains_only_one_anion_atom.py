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
def contains_only_one_anion_atom(self, anion_atom):
    """
        Whether this LightStructureEnvironments concerns a structure with only one given anion atom type.

        Args:
            anion_atom: Anion (e.g. O, ...). The structure could contain O2- and O- though.

        Returns:
            bool: True if this LightStructureEnvironments concerns a structure with only one given anion_atom.
        """
    return len(self.statistics_dict['anion_atom_list']) == 1 and anion_atom in self.statistics_dict['anion_atom_list']