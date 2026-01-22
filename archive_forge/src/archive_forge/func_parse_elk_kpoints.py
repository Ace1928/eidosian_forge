import collections
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
def parse_elk_kpoints(fd):
    header = next(fd)
    lhs, rhs = header.split(':', 1)
    assert 'k-point, vkl, wkpt' in rhs, header
    nkpts = int(lhs.strip())
    kpts = np.empty((nkpts, 3))
    weights = np.empty(nkpts)
    for ikpt in range(nkpts):
        line = next(fd)
        tokens = line.split()
        kpts[ikpt] = np.array(tokens[1:4]).astype(float)
        weights[ikpt] = float(tokens[4])
    yield ('ibz_kpoints', kpts)
    yield ('kpoint_weights', weights)