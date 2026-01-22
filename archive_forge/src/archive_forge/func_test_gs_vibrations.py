import numpy as np
from scipy.optimize import check_grad
from ase import Atoms
from ase.vibrations import Vibrations
from ase.calculators.morse import MorsePotential, fcut, fcut_d
from ase.build import bulk
def test_gs_vibrations(testdir):
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, Re]])
    atoms.calc = MorsePotential(epsilon=De, r0=Re, rho0=rho0)
    vib = Vibrations(atoms)
    vib.run()