from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def update_nb_set_environments(self, se, isite, cn, inb_set, nb_set, recompute=False, optimization=None):
    """
        Args:
            se:
            isite:
            cn:
            inb_set:
            nb_set:
            recompute:
            optimization:
        """
    ce = se.get_coordination_environments(isite=isite, cn=cn, nb_set=nb_set)
    if ce is not None and (not recompute):
        return ce
    ce = ChemicalEnvironments()
    neighb_coords = nb_set.neighb_coordsOpt if optimization == 2 else nb_set.neighb_coords
    self.setup_local_geometry(isite, coords=neighb_coords, optimization=optimization)
    if optimization > 0:
        logging.debug('Getting StructureEnvironments with optimized algorithm')
        nb_set.local_planes = {}
        nb_set.separations = {}
        cncgsm = self.get_coordination_symmetry_measures_optim(nb_set=nb_set, optimization=optimization)
    else:
        logging.debug('Getting StructureEnvironments with standard algorithm')
        cncgsm = self.get_coordination_symmetry_measures()
    for coord_geom_symb, dct in cncgsm.items():
        other_csms = {'csm_wocs_ctwocc': dct['csm_wocs_ctwocc'], 'csm_wocs_ctwcc': dct['csm_wocs_ctwcc'], 'csm_wocs_csc': dct['csm_wocs_csc'], 'csm_wcs_ctwocc': dct['csm_wcs_ctwocc'], 'csm_wcs_ctwcc': dct['csm_wcs_ctwcc'], 'csm_wcs_csc': dct['csm_wcs_csc'], 'rotation_matrix_wocs_ctwocc': dct['rotation_matrix_wocs_ctwocc'], 'rotation_matrix_wocs_ctwcc': dct['rotation_matrix_wocs_ctwcc'], 'rotation_matrix_wocs_csc': dct['rotation_matrix_wocs_csc'], 'rotation_matrix_wcs_ctwocc': dct['rotation_matrix_wcs_ctwocc'], 'rotation_matrix_wcs_ctwcc': dct['rotation_matrix_wcs_ctwcc'], 'rotation_matrix_wcs_csc': dct['rotation_matrix_wcs_csc'], 'scaling_factor_wocs_ctwocc': dct['scaling_factor_wocs_ctwocc'], 'scaling_factor_wocs_ctwcc': dct['scaling_factor_wocs_ctwcc'], 'scaling_factor_wocs_csc': dct['scaling_factor_wocs_csc'], 'scaling_factor_wcs_ctwocc': dct['scaling_factor_wcs_ctwocc'], 'scaling_factor_wcs_ctwcc': dct['scaling_factor_wcs_ctwcc'], 'scaling_factor_wcs_csc': dct['scaling_factor_wcs_csc'], 'translation_vector_wocs_ctwocc': dct['translation_vector_wocs_ctwocc'], 'translation_vector_wocs_ctwcc': dct['translation_vector_wocs_ctwcc'], 'translation_vector_wocs_csc': dct['translation_vector_wocs_csc'], 'translation_vector_wcs_ctwocc': dct['translation_vector_wcs_ctwocc'], 'translation_vector_wcs_ctwcc': dct['translation_vector_wcs_ctwcc'], 'translation_vector_wcs_csc': dct['translation_vector_wcs_csc']}
        ce.add_coord_geom(coord_geom_symb, dct['csm'], algo=dct['algo'], permutation=dct['indices'], local2perfect_map=dct['local2perfect_map'], perfect2local_map=dct['perfect2local_map'], detailed_voronoi_index={'cn': cn, 'index': inb_set}, other_symmetry_measures=other_csms, rotation_matrix=dct['rotation_matrix'], scaling_factor=dct['scaling_factor'])
    se.update_coordination_environments(isite=isite, cn=cn, nb_set=nb_set, ce=ce)
    return ce