from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
def standardize_structure(structure, sym_prec=0.1, international_monoclinic=True):
    """
    Get the symmetrically standardized structure.

    Args:
        structure (Structure): The structure.
        sym_prec (float): Tolerance for symmetry finding for standardization.
        international_monoclinic (bool): Whether to use international
            convention (vs Curtarolo) for monoclinic. Defaults True.

    Returns:
        The symmetrized structure.
    """
    sym_finder = SpacegroupAnalyzer(structure, symprec=sym_prec)
    new_structure = sym_finder.get_primitive_standard_structure(international_monoclinic=international_monoclinic)
    vpa_old = structure.volume / len(structure)
    vpa_new = new_structure.volume / len(new_structure)
    if abs(vpa_old - vpa_new) / vpa_old > 0.02:
        raise ValueError(f'Standardizing cell failed! VPA old: {vpa_old}, VPA new: {vpa_new}')
    matcher = StructureMatcher()
    if not matcher.fit(structure, new_structure):
        raise ValueError("Standardizing cell failed! Old structure doesn't match new.")
    return new_structure