import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def read_block(name, symlen, nvalues):
    """Read a data block.

            name: name of the block to store in self.data
            symlen: length of the symbol
            nvalues: number of values expected
            """
    if name not in self.data:
        self.data[name] = {}
    data = self.data[name]

    def add_line():
        line = fileobj.readline().strip()
        if not len(line):
            return False
        line = line.split('#')[0]
        if len(line) > symlen:
            symbol = line[:symlen]
            words = line[symlen:].split()
            if len(words) >= nvalues:
                if nvalues == 1:
                    data[symbol] = float(words[0])
                else:
                    data[symbol] = [float(word) for word in words[:nvalues]]
        return True
    while add_line():
        pass