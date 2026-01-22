import numpy as np
import ase.units
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.data import chemical_symbols
def read_acemolecule_out(filename):
    """Interface to ACEMoleculeReader and return values for corresponding quantity
    Parameters
    ==========
    filename: ACE-Molecule log file.
    quantity: One of atoms, energy, forces, excitation-energy.

    Returns
    =======
     - quantity = 'excitation-energy':
       returns None. This is placeholder function to run TDDFT calculations
       without IndexError.
     - quantity = 'energy':
       returns energy as float value.
     - quantity = 'forces':
       returns force of each atoms as numpy array of shape (natoms, 3).
     - quantity = 'atoms':
       returns ASE atoms object.
    """
    data = parse_geometry(filename)
    atom_symbol = np.array(data['Atomic_numbers'])
    positions = np.array(data['Positions'])
    atoms = Atoms(atom_symbol, positions=positions)
    energy = None
    forces = None
    excitation_energy = None
    with open(filename, 'r') as fd:
        lines = fd.readlines()
    for i in range(len(lines) - 1, 1, -1):
        line = lines[i].split()
        if len(line) > 2:
            if line[0] == 'Total' and line[1] == 'energy':
                energy = float(line[3])
                break
    energy *= ase.units.Hartree
    forces = []
    for i in range(len(lines) - 1, 1, -1):
        if '!============================' in lines[i]:
            endline_num = i
        if '! Force:: List of total force in atomic unit' in lines[i]:
            startline_num = i + 2
            for j in range(startline_num, endline_num):
                forces.append(lines[j].split()[3:6])
            convert = ase.units.Hartree / ase.units.Bohr
            forces = np.array(forces, dtype=float) * convert
            break
    if not len(forces) > 0:
        forces = None
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.calc = calc
    results = {}
    results['energy'] = energy
    results['atoms'] = atoms
    results['forces'] = forces
    results['excitation-energy'] = excitation_energy
    return results