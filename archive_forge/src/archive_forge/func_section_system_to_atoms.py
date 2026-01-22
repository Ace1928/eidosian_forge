import json
import numpy as np
import ase.units as units
from ase import Atoms
from ase.data import chemical_symbols
def section_system_to_atoms(section):
    """Covnert section_system into an Atoms object."""
    assert section['name'] == 'section_system'
    numbers = section['atom_species']
    numbers = np.array(numbers, int)
    numbers[numbers < 0] = 0
    numbers[numbers >= len(chemical_symbols)] = 0
    positions = section['atom_positions']['flatData']
    positions = np.array(positions).reshape(-1, 3) * units.m
    atoms = Atoms(numbers, positions=positions)
    atoms.info['nomad_uri'] = section['uri']
    pbc = section.get('configuration_periodic_dimensions')
    if pbc is not None:
        assert len(pbc) == 1
        pbc = pbc[0]
        pbc = pbc['flatData']
        assert len(pbc) == 3
        atoms.pbc = pbc
    cell = section.get('lattice_vectors')
    if cell is not None:
        cell = cell['flatData']
        cell = np.array(cell).reshape(3, 3) * units.m
        atoms.cell = cell
    return atoms