from __future__ import annotations
import copy
import os
import warnings
from itertools import groupby
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.compatibility import (
from pymatgen.entries.computed_entries import ComputedStructureEntry, ConstantEnergyAdjustment
from pymatgen.entries.entry_tools import EntrySet
Helper function to get spacegroup with a loose tolerance.