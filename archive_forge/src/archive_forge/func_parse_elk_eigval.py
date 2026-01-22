import collections
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
def parse_elk_eigval(fd):

    def match_int(line, word):
        number, colon, word1 = line.split()
        assert word1 == word
        assert colon == ':'
        return int(number)

    def skip_spaces(line=''):
        while not line.strip():
            line = next(fd)
        return line
    line = skip_spaces()
    nkpts = match_int(line, 'nkpt')
    line = next(fd)
    nbands = match_int(line, 'nstsv')
    eigenvalues = np.empty((nkpts, nbands))
    occupations = np.empty((nkpts, nbands))
    kpts = np.empty((nkpts, 3))
    for ikpt in range(nkpts):
        line = skip_spaces()
        tokens = line.split()
        assert tokens[-1] == 'vkl', tokens
        assert ikpt + 1 == int(tokens[0])
        kpts[ikpt] = np.array(tokens[1:4]).astype(float)
        line = next(fd)
        assert line.strip().startswith('(state,'), line
        for iband in range(nbands):
            line = next(fd)
            tokens = line.split()
            assert iband + 1 == int(tokens[0])
            eigenvalues[ikpt, iband] = float(tokens[1])
            occupations[ikpt, iband] = float(tokens[2])
    yield ('ibz_kpoints', kpts)
    yield ('eigenvalues', eigenvalues[None] * Hartree)
    yield ('occupations', occupations[None])