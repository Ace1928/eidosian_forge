import pytest
from ase.atoms import Atoms
def check_potcar(setups, filename='POTCAR'):
    """Return true if labels in setups are found in POTCAR"""
    pp = []
    with open(filename, 'r') as fd:
        for line in fd:
            if 'TITEL' in line.split():
                pp.append(line.split()[3])
    for setup in setups:
        assert setup in pp