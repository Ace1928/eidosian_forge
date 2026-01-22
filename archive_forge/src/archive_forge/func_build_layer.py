import pytest
import ase.build
from ase.io import write
def build_layer():
    atoms = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
    atoms.cell[2, 2] = 7
    atoms.set_pbc((1, 1, 1))
    return atoms