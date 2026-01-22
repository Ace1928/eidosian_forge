from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MontyDecoder
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

        Args:
            entry1: Entry corresponding to one of the entries in the voltage step.
            entry2: Entry corresponding to the other entry in the voltage step.
            working_ion_entry: A single ComputedEntry or PDEntry representing
                the element that carries charge across the battery, e.g. Li.
        