from the a CostDB instance, for example a CSV file via CostDBCSV.
from __future__ import annotations
import abc
import csv
import itertools
import os
from collections import defaultdict
import scipy.constants as const
from monty.design_patterns import singleton
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.util.provenance import is_valid_bibtex
def get_cost_per_mol(self, comp):
    """
        Get best estimate of minimum cost/mol based on known data.

        Args:
            comp:
                Composition as a pymatgen.core.structure.Composition

        Returns:
            float of cost/mol
        """
    comp = comp if isinstance(comp, Composition) else Composition(comp)
    decomp = self.get_lowest_decomposition(comp)
    return sum((k.energy_per_atom * v * comp.num_atoms for k, v in decomp.items()))