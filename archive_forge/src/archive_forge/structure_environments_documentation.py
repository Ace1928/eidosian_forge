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

            Reconstructs the NeighborsSet algorithm from its JSON-serializable dict representation, together with
            the structure and all the possible neighbors sites.

            As an inner (nested) class, the NeighborsSet is not supposed to be used anywhere else that inside the
            LightStructureEnvironments. The from_dict method is thus using the structure and all_nbs_sites when
            reconstructing itself. These two are both in the LightStructureEnvironments object.

            Args:
                dct: a JSON-serializable dict representation of a NeighborsSet.
                structure: The structure.
                all_nbs_sites: The list of all the possible neighbors for a given site.

            Returns:
                NeighborsSet
            