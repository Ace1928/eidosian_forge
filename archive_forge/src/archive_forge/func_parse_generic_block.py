import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def parse_generic_block(block):
    """
            Parse any other block into data dictionary given list of record
            tuples.
        """
    name, records = block
    data_dict = {}
    for record in records:
        tag, data = record
        if tag not in data_dict:
            data_dict[tag] = []
        data_dict[tag].append(data)
    return data_dict