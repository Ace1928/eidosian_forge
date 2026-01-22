import numpy as np
from collections import namedtuple
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality import topology_scaling
from ase.geometry.dimensionality.bond_generator import next_bond
def build_kinterval(a, b, h, components, cdim, score=None):
    if score is None:
        score = calculate_score(a, b)
    return Kinterval(dimtype=build_dimtype(h), score=score, a=a, b=b, h=h, components=components, cdim=cdim)