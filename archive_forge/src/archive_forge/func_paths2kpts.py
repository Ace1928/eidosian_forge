import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def paths2kpts(paths, cell, npoints=None, density=None):
    if not (npoints is None or density is None):
        raise ValueError('You may define npoints or density, but not both.')
    points = np.concatenate(paths)
    dists = points[1:] - points[:-1]
    lengths = [np.linalg.norm(d) for d in kpoint_convert(cell, skpts_kc=dists)]
    i = 0
    for path in paths[:-1]:
        i += len(path)
        lengths[i - 1] = 0
    length = sum(lengths)
    if npoints is None:
        if density is None:
            density = DEFAULT_KPTS_DENSITY
        npoints = int(round(length * density))
    kpts = []
    x0 = 0
    x = []
    X = [0]
    for P, d, L in zip(points[:-1], dists, lengths):
        diff = length - x0
        if abs(diff) < 1e-06:
            n = 0
        else:
            n = max(2, int(round(L * (npoints - len(x)) / diff)))
        for t in np.linspace(0, 1, n)[:-1]:
            kpts.append(P + t * d)
            x.append(x0 + t * L)
        x0 += L
        X.append(x0)
    if len(points):
        kpts.append(points[-1])
        x.append(x0)
    if len(kpts) == 0:
        kpts = np.empty((0, 3))
    return (np.array(kpts), np.array(x), np.array(X))