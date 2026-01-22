import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def parse_blocks(file_contents):
    """
            Parse series of XML-like deliminated blocks into a list of
            (block_name, contents) tuples
        """
    blocks = blocks_re.findall(file_contents)
    return blocks