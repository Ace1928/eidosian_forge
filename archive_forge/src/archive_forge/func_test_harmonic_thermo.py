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
def test_harmonic_thermo(testdir):
    atoms = fcc100('Cu', (2, 2, 2), vacuum=10.0)
    atoms.calc = EMT()
    add_adsorbate(atoms, 'Pt', 1.5, 'hollow')
    atoms.set_constraint(FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Cu']))
    QuasiNewton(atoms).run(fmax=0.01)
    vib = Vibrations(atoms, name='harmonicthermo-vib', indices=[atom.index for atom in atoms if atom.symbol != 'Cu'])
    vib.run()
    vib.summary()
    vib_energies = vib.get_energies()
    thermo = HarmonicThermo(vib_energies=vib_energies, potentialenergy=atoms.get_potential_energy())
    thermo.get_helmholtz_energy(temperature=298.15)