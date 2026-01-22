from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
def get_orb_from_str(orbs):
    """

    Args:
        orbs: list of two or more str, e.g. ["2p_x", "3s"].

    Returns:
        list of tw Orbital objects
    """
    orb_labs = ['s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z^2', 'd_xz', 'd_x^2-y^2', 'f_y(3x^2-y^2)', 'f_xyz', 'f_yz^2', 'f_z^3', 'f_xz^2', 'f_z(x^2-y^2)', 'f_x(x^2-3y^2)']
    orbitals = [(int(orb[0]), Orbital(orb_labs.index(orb[1:]))) for orb in orbs]
    orb_label = ''
    for iorb, orbital in enumerate(orbitals):
        if iorb == 0:
            orb_label += f'{orbital[0]}{orbital[1].name}'
        else:
            orb_label += f'-{orbital[0]}{orbital[1].name}'
    return (orb_label, orbitals)