import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def parse_block(block):
    """
            Parse block contents into a series of (tag, data) records
        """

    def clean_line(line):
        line = re.sub('#(.*?)\n', '', line)
        line = line.strip()
        return line
    name, data = block
    lines = [clean_line(line) for line in data.split('\n')]
    records = []
    for line in lines:
        xs = line.split()
        if len(xs) > 0:
            tag = xs[0]
            data = xs[1:]
            records.append((tag, data))
    return (name, records)