from __future__ import annotations
import copy
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.inputs import AimsControlIn, AimsGeometryIn
from pymatgen.io.aims.parsers import AimsParseError, read_aims_output
from pymatgen.io.core import InputFile, InputGenerator, InputSet
@staticmethod
def d2k_recipcell(recipcell: np.ndarray, pbc: Sequence[bool], kptdensity: float | Sequence[float]=5.0, even: bool=True) -> Sequence[int]:
    """Convert k-point density to Monkhorst-Pack grid size.

        Parameters
        ----------
        recipcell: Cell
            The reciprocal cell
        pbc: Sequence[bool]
            If element of pbc is True then system is periodic in that direction
        kptdensity: float or list[floats]
            Required k-point density.  Default value is 3.5 point per Ang^-1.
        even: bool
            Round up to even numbers.

        Returns:
            dict: Monkhorst-Pack grid size in all directions
        """
    if not isinstance(kptdensity, Iterable):
        kptdensity = 3 * [float(kptdensity)]
    kpts: list[int] = []
    for i in range(3):
        if pbc[i]:
            k = 2 * np.pi * np.sqrt((recipcell[i] ** 2).sum()) * float(kptdensity[i])
            if even:
                kpts.append(2 * int(np.ceil(k / 2)))
            else:
                kpts.append(int(np.ceil(k)))
        else:
            kpts.append(1)
    return kpts