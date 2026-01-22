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
def process_entry(self, entry: ComputedEntry, **kwargs) -> ComputedEntry | None:
    """Process a single entry with the chosen Corrections. Note
        that this method will change the data of the original entry.

        Args:
            entry: A ComputedEntry object.
            **kwargs: Will be passed to process_entries().

        Returns:
            An adjusted entry if entry is compatible, else None.
        """
    try:
        return self.process_entries(entry, **kwargs)[0]
    except IndexError:
        return None