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
def has_good_quality_check_occupied_bands(self, number_occ_bands_spin_up: int, number_occ_bands_spin_down: int | None=None, spin_polarized: bool=False, limit_deviation: float=0.1) -> bool:
    """
        Will check if the deviation from the ideal bandoverlap of all occupied bands
        is smaller or equal to limit_deviation.

        Args:
            number_occ_bands_spin_up (int): number of occupied bands of spin up
            number_occ_bands_spin_down (int): number of occupied bands of spin down
            spin_polarized (bool): If True, then it was a spin polarized calculation
            limit_deviation (float): limit of the maxDeviation

        Returns:
            Boolean that will give you information about the quality of the projection
        """
    for matrix in self.band_overlaps_dict[Spin.up]['matrices']:
        for iband1, band1 in enumerate(matrix):
            for iband2, band2 in enumerate(band1):
                if iband1 < number_occ_bands_spin_up and iband2 < number_occ_bands_spin_up:
                    if iband1 == iband2:
                        if abs(band2 - 1.0).all() > limit_deviation:
                            return False
                    elif band2.all() > limit_deviation:
                        return False
    if spin_polarized:
        for matrix in self.band_overlaps_dict[Spin.down]['matrices']:
            for iband1, band1 in enumerate(matrix):
                for iband2, band2 in enumerate(band1):
                    if number_occ_bands_spin_down is not None:
                        if iband1 < number_occ_bands_spin_down and iband2 < number_occ_bands_spin_down:
                            if iband1 == iband2:
                                if abs(band2 - 1.0).all() > limit_deviation:
                                    return False
                            elif band2.all() > limit_deviation:
                                return False
                    else:
                        raise ValueError('number_occ_bands_spin_down has to be specified')
    return True