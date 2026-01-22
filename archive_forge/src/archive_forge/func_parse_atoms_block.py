import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def parse_atoms_block(block):
    """
            Parse atoms block into data dictionary given list of record tuples.
        """
    name, records = block

    def lattice(d):
        return tensor33([float(x) for x in data])

    def atom(d):
        return {'species': data[0], 'label': data[1], 'index': int(data[2]), 'position': tensor31([float(x) for x in data[3:]])}

    def symmetry(d):
        return ' '.join(data)
    tags = {'lattice': lattice, 'atom': atom, 'units': check_units, 'symmetry': symmetry}
    data_dict = {}
    for record in records:
        tag, data = record
        if tag not in data_dict:
            data_dict[tag] = []
        data_dict[tag].append(tags[tag](data))
    return data_dict