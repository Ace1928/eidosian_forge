from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_energy_tersoff(structure, gulp_cmd='gulp'):
    """Compute the energy of a structure using Tersoff potential.

    Args:
        structure: pymatgen Structure
        gulp_cmd: GULP command if not in standard place
    """
    gio = GulpIO()
    gc = GulpCaller(gulp_cmd)
    gin = gio.tersoff_input(structure)
    gout = gc.run(gin)
    return gio.get_energy(gout)