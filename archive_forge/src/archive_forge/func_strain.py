import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def strain(at, e, calc):
    at = at.copy()
    at.set_cell((1.0 + e) * at.cell, scale_atoms=True)
    at.calc = calc
    v = at.get_volume()
    e = at.get_potential_energy()
    return (v, e)