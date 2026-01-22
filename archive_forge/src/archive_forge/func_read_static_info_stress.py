import re
import numpy as np
from ase.units import Bohr, Angstrom, Hartree, eV, Debye
def read_static_info_stress(fd):
    stress_cv = np.empty((3, 3))
    headers = next(fd)
    assert headers.strip().startswith('T_{ij}')
    for i in range(3):
        line = next(fd)
        tokens = line.split()
        vec = np.array(tokens[1:4]).astype(float)
        stress_cv[i] = vec
    return stress_cv