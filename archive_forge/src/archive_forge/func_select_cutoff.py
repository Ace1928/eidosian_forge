import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
def select_cutoff(atoms):
    intervals = analyze_dimensionality(atoms, method='RDA', merge=False)
    dimtype = max(merge_intervals(intervals), key=lambda x: x.score).dimtype
    m = next((e for e in intervals if e.dimtype == dimtype))
    if m.b == float('inf'):
        return m.a + 0.1
    else:
        return (m.a + m.b) / 2