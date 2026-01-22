import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
Read a data block.

            name: name of the block to store in self.data
            symlen: length of the symbol
            nvalues: number of values expected
            