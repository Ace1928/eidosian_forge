import os
import os.path as op
import subprocess
import shutil
import numpy as np
from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.calculator import equal
import ase.io
from .demon_io import parse_xray
def read_eigenvalues_one_spin(self, lines, string, neigs_per_line):
    """Utility method for retreiving eigenvalues after the string "string"
        with neigs_per_line eigenvlaues written per line
        """
    eig = []
    occ = []
    skip_line = False
    more_eigs = False
    for i in range(len(lines)):
        if lines[i].rfind(string) > -1:
            ii = i
            more_eigs = True
            break
    while more_eigs:
        for i in range(ii + 1, len(lines)):
            if len(lines[i].split()) == 0 and len(lines[i + 1].split()) == 0 and (len(lines[i + 2].split()) > 0):
                ii = i + 2
                break
        line = lines[ii].split()
        if len(line) < neigs_per_line:
            more_eigs = False
        if line[0] != str(len(eig) + 1):
            more_eigs = False
            skip_line = True
        if not skip_line:
            line = lines[ii + 1].split()
            for l in line:
                eig.append(float(l))
            line = lines[ii + 3].split()
            for l in line:
                occ.append(float(l))
            ii = ii + 3
    return (eig, occ)