import numpy as np
from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.build import bulk
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.phonons import Phonons
from ase.thermochemistry import (IdealGasThermo, HarmonicThermo,
from ase.calculators.emt import EMT
def test_crystal_thermo(asap3, testdir):
    atoms = bulk('Al', 'fcc', a=4.05)
    calc = asap3.EMT()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    N = 7
    ph = Phonons(atoms, calc, supercell=(N, N, N), delta=0.05)
    ph.run()
    ph.read(acoustic=True)
    phonon_energies, phonon_DOS = ph.dos(kpts=(4, 4, 4), npts=30, delta=0.0005)
    thermo = CrystalThermo(phonon_energies=phonon_energies, phonon_DOS=phonon_DOS, potentialenergy=energy, formula_units=4)
    thermo.get_helmholtz_energy(temperature=298.15)