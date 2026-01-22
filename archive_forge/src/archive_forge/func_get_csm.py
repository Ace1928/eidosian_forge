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
def get_csm(self, isite, mp_symbol):
    """
        Get the continuous symmetry measure for a given site in the given coordination environment.

        Args:
            isite: Index of the site.
            mp_symbol: Symbol of the coordination environment for which we want the continuous symmetry measure.

        Returns:
            Continuous symmetry measure of the given site in the given environment.
        """
    csms = self.get_csms(isite, mp_symbol)
    if len(csms) != 1:
        raise ChemenvError('StructureEnvironments', 'get_csm', f'Number of csms for site #{isite} with mp_symbol {mp_symbol!r} = {len(csms)}')
    return csms[0]