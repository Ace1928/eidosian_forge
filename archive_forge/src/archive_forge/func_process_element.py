import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
def process_element(self, element):
    """Extract atomic number from element"""
    if self.element_basis is None:
        if isinstance(element, str):
            self.atomicnumber = ase.data.atomic_numbers[element]
        elif isinstance(element, int):
            self.atomicnumber = element
        else:
            raise TypeError('The symbol argument must be a string or an atomic number.')
    else:
        atomicnumber = []
        try:
            if len(element) != max(self.element_basis) + 1:
                oops = True
            else:
                oops = False
        except TypeError:
            oops = True
        if oops:
            raise TypeError(('The symbol argument must be a sequence of length %d' + ' (one for each kind of lattice position') % (max(self.element_basis) + 1,))
        for e in element:
            if isinstance(e, str):
                atomicnumber.append(ase.data.atomic_numbers[e])
            elif isinstance(e, int):
                atomicnumber.append(e)
            else:
                raise TypeError('The symbols argument must be a sequence of strings or atomic numbers.')
        self.atomicnumber = [atomicnumber[i] for i in self.element_basis]
        assert len(self.atomicnumber) == len(self.bravais_basis)