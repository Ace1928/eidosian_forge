import collections
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
def parse_elk_info(fd):
    dct = collections.defaultdict(list)
    fd = iter(fd)
    spinpol = None
    converged = False
    actually_did_not_converge = False
    for line in fd:
        tokens = line.split(':', 1)
        if len(tokens) == 2:
            lhs, rhs = tokens
            dct[lhs.strip()].append(rhs.strip())
        elif line.startswith('Convergence targets achieved'):
            converged = True
        elif 'reached self-consistent loops maximum' in line.lower():
            actually_did_not_converge = True
        if 'Spin treatment' in line:
            line = next(fd)
            spinpol = line.strip() == 'spin-polarised'
    yield ('converged', converged and (not actually_did_not_converge))
    if spinpol is None:
        raise RuntimeError('Could not determine spin treatment')
    yield ('spinpol', spinpol)
    if 'Fermi' in dct:
        yield ('fermi_level', float(dct['Fermi'][-1]) * Hartree)
    if 'total force' in dct:
        forces = []
        for line in dct['total force']:
            forces.append(line.split())
        yield ('forces', np.array(forces, float) * (Hartree / Bohr))