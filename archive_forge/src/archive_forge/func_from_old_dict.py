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
@classmethod
def from_old_dict(cls, dct) -> Self:
    """
        Args:
            dct (dict): A dict with all data for a band structure symmetry line
                object.

        Returns:
            A BandStructureSymmLine object
        """
    labels_dict = {k.strip(): v for k, v in dct['labels_dict'].items()}
    projections: dict = {}
    structure = None
    if 'projections' in dct and len(dct['projections']) != 0:
        structure = Structure.from_dict(dct['structure'])
        projections = {}
        for spin in dct['projections']:
            dd = []
            for i in range(len(dct['projections'][spin])):
                ddd = []
                for j in range(len(dct['projections'][spin][i])):
                    ddd.append(dct['projections'][spin][i][j])
                dd.append(np.array(ddd))
            projections[Spin(int(spin))] = np.array(dd)
    return cls(dct['kpoints'], {Spin(int(k)): dct['bands'][k] for k in dct['bands']}, Lattice(dct['lattice_rec']['matrix']), dct['efermi'], labels_dict, structure=structure, projections=projections)