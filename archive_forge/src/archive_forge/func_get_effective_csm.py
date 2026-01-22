from __future__ import annotations
import abc
import os
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from monty.json import MSONable
from scipy.stats import gmean
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import EquivalentSiteSearchError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import get_lower_and_upper_f
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.func_utils import (
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_effective_csm(nb_set, cn_map, structure_environments, additional_info, symmetry_measure_type, max_effective_csm, effective_csm_estimator_ratio_function):
    """Get the effective continuous symmetry measure of a given neighbors set.

    Args:
        nb_set: Neighbors set.
        cn_map: Mapping index of this neighbors set.
        structure_environments: Structure environments.
        additional_info: Additional information for the neighbors set.
        symmetry_measure_type: Type of symmetry measure to be used in the effective CSM.
        max_effective_csm: Max CSM to use for the effective CSM calculation.
        effective_csm_estimator_ratio_function: Ratio function to use to compute effective CSM.
    Returns:
        Effective CSM of a given Neighbors set.
    """
    try:
        effective_csm = additional_info['effective_csms'][nb_set.isite][cn_map]
    except KeyError:
        site_ce_list = structure_environments.ce_list[nb_set.isite]
        site_chemenv = site_ce_list[cn_map[0]][cn_map[1]]
        if site_chemenv is None:
            effective_csm = 100
        else:
            mingeoms = site_chemenv.minimum_geometries(symmetry_measure_type=symmetry_measure_type, max_csm=max_effective_csm)
            if len(mingeoms) == 0:
                effective_csm = 100
            else:
                csms = [ce_dict['other_symmetry_measures'][symmetry_measure_type] for mp_symbol, ce_dict in mingeoms if ce_dict['other_symmetry_measures'][symmetry_measure_type] <= max_effective_csm]
                effective_csm = effective_csm_estimator_ratio_function.mean_estimator(csms)
        set_info(additional_info=additional_info, field='effective_csms', isite=nb_set.isite, cn_map=cn_map, value=effective_csm)
    return effective_csm