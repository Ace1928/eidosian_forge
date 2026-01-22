from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from tabulate import tabulate
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.core import ParseError
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
@staticmethod
def pot_string_from_file(filename='feff.inp'):
    """
        Reads Potential parameters from a feff.inp or FEFFPOT file.
        The lines are arranged as follows:

          ipot   Z   element   lmax1   lmax2   stoichometry   spinph

        Args:
            filename: file name containing potential data.

        Returns:
            FEFFPOT string.
        """
    with zopen(filename, mode='rt') as f_object:
        f = f_object.readlines()
        ln = -1
        pot_str = ['POTENTIALS\n']
        pot_tag = -1
        pot_data = 0
        pot_data_over = 1
        sep_line_pattern = [re.compile('ipot.*Z.*tag.*lmax1.*lmax2.*spinph'), re.compile('^[*]+.*[*]+$')]
        for line in f:
            if pot_data_over == 1:
                ln += 1
                if pot_tag == -1:
                    pot_tag = line.find('POTENTIALS')
                    ln = 0
                if pot_tag >= 0 and ln > 0 and (pot_data_over > 0):
                    try:
                        if len(sep_line_pattern[0].findall(line)) > 0 or len(sep_line_pattern[1].findall(line)) > 0:
                            pot_str.append(line)
                        elif int(line.split()[0]) == pot_data:
                            pot_data += 1
                            pot_str.append(line.replace('\r', ''))
                    except (ValueError, IndexError):
                        if pot_data > 0:
                            pot_data_over = 0
    return ''.join(pot_str).rstrip('\n')