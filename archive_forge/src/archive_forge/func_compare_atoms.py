import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
def compare_atoms(atoms1, atoms2, tol=1e-15, excluded_properties=None):
    """Check for system changes since last calculation.  Properties in
    ``excluded_properties`` are not checked."""
    if atoms1 is None:
        system_changes = all_changes[:]
    else:
        system_changes = []
        properties_to_check = set(all_changes)
        if excluded_properties:
            properties_to_check -= set(excluded_properties)
        for prop in ['cell', 'pbc']:
            if prop in properties_to_check:
                properties_to_check.remove(prop)
                if not equal(getattr(atoms1, prop), getattr(atoms2, prop), atol=tol):
                    system_changes.append(prop)
        arrays1 = set(atoms1.arrays)
        arrays2 = set(atoms2.arrays)
        system_changes += properties_to_check & (arrays1 ^ arrays2)
        for prop in properties_to_check & arrays1 & arrays2:
            if not equal(atoms1.arrays[prop], atoms2.arrays[prop], atol=tol):
                system_changes.append(prop)
    return system_changes