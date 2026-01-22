from __future__ import annotations
import copy
import logging
from ast import literal_eval
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.json import MSONable, jsanitize
from monty.serialization import dumpfn
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer, Ordering
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_low_energy_orderings(self):
    """
        Find lowest energy FM and AFM orderings to compute E_AFM - E_FM.

        Returns:
            fm_struct (Structure): fm structure with 'magmom' site property
            afm_struct (Structure): afm structure with 'magmom' site property
            fm_e (float): fm energy
            afm_e (float): afm energy
        """
    fm_struct, afm_struct = (None, None)
    mag_min = np.inf
    mag_max = 0.001
    fm_e_min = 0
    afm_e_min = 0
    for s, e in zip(self.ordered_structures, self.energies):
        ordering = CollinearMagneticStructureAnalyzer(s, threshold=0, make_primitive=False).ordering
        magmoms = s.site_properties['magmom']
        if ordering == Ordering.FM and e < fm_e_min:
            fm_struct = s
            mag_max = abs(sum(magmoms))
            fm_e = e
            fm_e_min = e
        if ordering == Ordering.AFM and e < afm_e_min:
            afm_struct = s
            afm_e = e
            mag_min = abs(sum(magmoms))
            afm_e_min = e
    if not fm_struct or not afm_struct:
        for s, e in zip(self.ordered_structures, self.energies):
            magmoms = s.site_properties['magmom']
            if abs(sum(magmoms)) > mag_max:
                fm_struct = s
                fm_e = e
                mag_max = abs(sum(magmoms))
            if abs(sum(magmoms)) < mag_min:
                afm_struct = s
                afm_e = e
                mag_min = abs(sum(magmoms))
                afm_e_min = e
            elif abs(sum(magmoms)) == 0 and mag_min == 0 and (e < afm_e_min):
                afm_struct = s
                afm_e = e
                afm_e_min = e
    fm_struct = CollinearMagneticStructureAnalyzer(fm_struct, make_primitive=False, threshold=0.0).get_structure_with_only_magnetic_atoms(make_primitive=False)
    afm_struct = CollinearMagneticStructureAnalyzer(afm_struct, make_primitive=False, threshold=0.0).get_structure_with_only_magnetic_atoms(make_primitive=False)
    return (fm_struct, afm_struct, fm_e, afm_e)