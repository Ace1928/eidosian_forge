import io
import re
import pathlib
import numpy as np
from ase.calculators.lammps import convert
def extract_mass(raw_datafile_contents):
    """
    NOTE: Assumes that only a single atomic species is present
    """
    masses_block = extract_section(raw_datafile_contents, 'Masses')
    if masses_block is None:
        return None
    else:
        mass = re.match('\\s*[0-9]+\\s+(\\S+)', masses_block).group(1)
        return float(mass)