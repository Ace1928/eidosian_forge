from __future__ import annotations
import collections
import itertools
import math
import re
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff
def get_vbm(self):
    """Returns data about the VBM.

        Returns:
            dict: With keys "band_index", "kpoint_index", "kpoint", "energy"
            - "band_index": A dict with spin keys pointing to a list of the
            indices of the band containing the VBM (please note that you
            can have several bands sharing the VBM) {Spin.up:[],
            Spin.down:[]}
            - "kpoint_index": The list of indices in self.kpoints for the
            kpoint VBM. Please note that there can be several
            kpoint_indices relating to the same kpoint (e.g., Gamma can
            occur at different spots in the band structure line plot)
            - "kpoint": The kpoint (as a kpoint object)
            - "energy": The energy of the VBM
            - "projections": The projections along sites and orbitals of the
            VBM if any projection data is available (else it is an empty
            dictionary). The format is similar to the projections field in
            BandStructure: {spin:{'Orbital': [proj]}} where the array
            [proj] is ordered according to the sites in structure
        """
    if self.is_metal():
        return {'band_index': [], 'kpoint_index': [], 'kpoint': [], 'energy': None, 'projections': {}}
    max_tmp = -float('inf')
    index = kpoint_vbm = None
    for value in self.bands.values():
        for i, j in zip(*np.where(value < self.efermi)):
            if value[i, j] > max_tmp:
                max_tmp = float(value[i, j])
                index = j
                kpoint_vbm = self.kpoints[j]
    list_ind_kpts = []
    if kpoint_vbm.label is not None:
        for i, kpt in enumerate(self.kpoints):
            if kpt.label == kpoint_vbm.label:
                list_ind_kpts.append(i)
    else:
        list_ind_kpts.append(index)
    list_ind_band = collections.defaultdict(list)
    for spin in self.bands:
        for i in range(self.nb_bands):
            if math.fabs(self.bands[spin][i][index] - max_tmp) < 0.001:
                list_ind_band[spin].append(i)
    proj = {}
    for spin, value in self.projections.items():
        if len(list_ind_band[spin]) == 0:
            continue
        proj[spin] = value[list_ind_band[spin][0]][list_ind_kpts[0]]
    return {'band_index': list_ind_band, 'kpoint_index': list_ind_kpts, 'kpoint': kpoint_vbm, 'energy': max_tmp, 'projections': proj}