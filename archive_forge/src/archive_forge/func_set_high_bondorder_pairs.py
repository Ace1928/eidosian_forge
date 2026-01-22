from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
def set_high_bondorder_pairs(bondpairs, high_bondorder_pairs=None):
    """Set high bondorder pairs

    Modify bondpairs list (from get_bondpairs((atoms)) to include high
    bondorder pairs.

    Parameters:
    -----------
    bondpairs: List of pairs, generated from get_bondpairs(atoms)
    high_bondorder_pairs: Dictionary of pairs with high bond orders
                          using the following format:
                          { ( a1, b1 ): ( offset1, bond_order1, bond_offset1),
                            ( a2, b2 ): ( offset2, bond_order2, bond_offset2),
                            ...
                          }
                          offset, bond_order, bond_offset are optional.
                          However, if they are provided, the 1st value is
                          offset, 2nd value is bond_order,
                          3rd value is bond_offset """
    if high_bondorder_pairs is None:
        high_bondorder_pairs = dict()
    bondpairs_ = []
    for pair in bondpairs:
        a, b = (pair[0], pair[1])
        if (a, b) in high_bondorder_pairs.keys():
            bondpair = [a, b] + [item for item in high_bondorder_pairs[a, b]]
            bondpairs_.append(bondpair)
        elif (b, a) in high_bondorder_pairs.keys():
            bondpair = [a, b] + [item for item in high_bondorder_pairs[b, a]]
            bondpairs_.append(bondpair)
        else:
            bondpairs_.append(pair)
    return bondpairs_