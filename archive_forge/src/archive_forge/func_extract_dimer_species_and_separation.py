import pytest
import numpy as np
from ase import Atoms
def extract_dimer_species_and_separation(atoms):
    """
    Given a monoatomic dimer, extract the species of its atoms and their
    separation
    """
    if len(set(atoms.symbols)) > 1:
        raise ValueError('Dimer must contain only one atomic species')
    spec = atoms.symbols[0]
    pos = atoms.get_positions()
    a = np.linalg.norm(pos[1] - pos[0])
    return (spec, a)