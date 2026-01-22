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
def test_ideal_gas_thermo(testdir):
    atoms = Atoms('N2', positions=[(0, 0, 0), (0, 0, 1.1)])
    atoms.calc = EMT()
    QuasiNewton(atoms).run(fmax=0.01)
    energy = atoms.get_potential_energy()
    vib = Vibrations(atoms, name='idealgasthermo-vib')
    vib.run()
    vib_energies = vib.get_energies()
    thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear', atoms=atoms, symmetrynumber=2, spin=0, potentialenergy=energy)
    thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.0)