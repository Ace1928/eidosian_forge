from __future__ import annotations
import abc
import copy
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Union
import numpy as np
from monty.design_patterns import cached_class
from monty.json import MSONable
from monty.serialization import loadfn
from tqdm import tqdm
from uncertainties import ufloat
from pymatgen.analysis.structure_analyzer import oxide_type, sulfide_type
from pymatgen.core import SETTINGS, Composition, Element
from pymatgen.entries.computed_entries import (
from pymatgen.io.vasp.sets import MITRelaxSet, MPRelaxSet, VaspInputSet
from pymatgen.util.due import Doi, due
def correct_entry(self, entry):
    """Corrects a single entry.

        Args:
            entry: A ComputedEntry object.

        Returns:
            An processed entry.

        Raises:
            CompatibilityError if entry is not compatible.
        """
    new_corr = self.get_correction(entry)
    old_std_dev = entry.correction_uncertainty
    if np.isnan(old_std_dev):
        old_std_dev = 0
    old_corr = ufloat(entry.correction, old_std_dev)
    updated_corr = new_corr + old_corr
    uncertainty = np.nan if updated_corr.nominal_value != 0 and updated_corr.std_dev == 0 else updated_corr.std_dev
    entry.energy_adjustments.append(ConstantEnergyAdjustment(updated_corr.nominal_value, uncertainty))
    return entry