from __future__ import annotations
import collections
import functools
import operator
import os
from math import exp, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Element, Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def evaluate_assignment(v_set):
    el_oxi = collections.defaultdict(list)
    jj = 0
    for sites in equi_sites:
        for specie, _ in get_z_ordered_elmap(sites[0].species):
            el_oxi[specie.symbol].append(v_set[jj])
            jj += 1
    max_diff = max((max(v) - min(v) for v in el_oxi.values()))
    if max_diff > 2:
        return
    score = functools.reduce(operator.mul, [all_prob[attrib[iv]][elements[iv]][vv] for iv, vv in enumerate(v_set)])
    if score > self._best_score:
        self._best_vset = v_set
        self._best_score = score