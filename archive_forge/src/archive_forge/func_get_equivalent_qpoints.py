from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def get_equivalent_qpoints(self, index: int) -> list[int]:
    """Returns the list of qpoint indices equivalent (meaning they are the
        same frac coords) to the given one.

        Args:
            index (int): the qpoint index

        Returns:
            list[int]: equivalent indices

        TODO: now it uses the label we might want to use coordinates instead
        (in case there was a mislabel)
        """
    if self.qpoints[index].label is None:
        return [index]
    list_index_qpoints = []
    for idx in range(self.nb_qpoints):
        if self.qpoints[idx].label == self.qpoints[index].label:
            list_index_qpoints.append(idx)
    return list_index_qpoints