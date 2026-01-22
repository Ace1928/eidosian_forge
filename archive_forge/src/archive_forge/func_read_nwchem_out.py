import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def read_nwchem_out(fobj, index=-1):
    """Splits an NWChem output file into chunks corresponding to
    individual single point calculations."""
    lines = fobj.readlines()
    if index == slice(-1, None, None):
        for line in lines:
            if _gauss_block.match(line):
                return [parse_gto_chunk(''.join(lines))]
            if _pw_block.match(line):
                return [parse_pw_chunk(''.join(lines))]
        else:
            raise ValueError('This does not appear to be a valid NWChem output file.')
    group = []
    atomslist = []
    header = True
    lastgroup = []
    lastparser = None
    parser = None
    for line in lines:
        group.append(line)
        if _gauss_block.match(line):
            next_parser = parse_gto_chunk
        elif _pw_block.match(line):
            next_parser = parse_pw_chunk
        else:
            continue
        if header:
            header = False
        else:
            atoms = parser(''.join(group))
            if atoms is None and parser is lastparser:
                atoms = parser(''.join(lastgroup + group))
                if atoms is not None:
                    atomslist[-1] = atoms
                    lastgroup += group
            else:
                atomslist.append(atoms)
                lastgroup = group
                lastparser = parser
            group = []
        parser = next_parser
    else:
        if not header:
            atoms = parser(''.join(group))
            if atoms is not None:
                atomslist.append(atoms)
    return atomslist[index]