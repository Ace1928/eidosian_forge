import numpy as np
from ase.io.fortranfile import FortranFile
def readDIM(fname):
    """
    Read unformatted siesta DIM file
    """
    import collections
    DIM_tuple = collections.namedtuple('DIM', ['natoms_sc', 'norbitals_sc', 'norbitals', 'nspin', 'nnonzero', 'natoms_interacting'])
    fh = FortranFile(fname)
    natoms_sc = fh.readInts('i')[0]
    norbitals_sc = fh.readInts('i')[0]
    norbitals = fh.readInts('i')[0]
    nspin = fh.readInts('i')[0]
    nnonzero = fh.readInts('i')[0]
    natoms_interacting = fh.readInts('i')[0]
    fh.close()
    return DIM_tuple(natoms_sc, norbitals_sc, norbitals, nspin, nnonzero, natoms_interacting)