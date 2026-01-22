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
def get_projections_on_elements_and_orbitals(self, el_orb_spec):
    """Method returning a dictionary of projections on elements and specific
        orbitals.

        Args:
            el_orb_spec: A dictionary of Elements and Orbitals for which we want
                to have projections on. It is given as: {Element:[orbitals]},
                e.g., {'Si':['3s','3p']} or {'Si':['3s','3p_x', '3p_y', '3p_z']} depending on input files

        Returns:
            A dictionary of projections on elements in the
            {Spin.up:[][{Element:{orb:values}}],
            Spin.down:[][{Element:{orb:values}}]} format
            if there is no projections in the band structure returns an empty
            dict.
        """
    result = {}
    el_orb_spec = {get_el_sp(el): orbs for el, orbs in el_orb_spec.items()}
    for spin, v in self.projections.items():
        result[spin] = [[{str(e): collections.defaultdict(float) for e in el_orb_spec} for i in range(len(self.kpoints))] for j in range(self.nb_bands)]
        for i, j in itertools.product(range(self.nb_bands), range(len(self.kpoints))):
            for key, item in v[i][j].items():
                for key2, item2 in item.items():
                    specie = str(Element(re.split('[0-9]+', key)[0]))
                    if get_el_sp(str(specie)) in el_orb_spec and key2 in el_orb_spec[get_el_sp(str(specie))]:
                        result[spin][i][j][specie][key2] += item2
    return result