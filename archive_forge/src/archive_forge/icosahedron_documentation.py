import numpy as np
from ase import Atoms
from ase.cluster.util import get_element_info

    Returns a cluster with the icosahedra symmetry.

    Parameters
    ----------
    symbol: The chemical symbol (or atomic number) of the element.

    noshells: The number of shells (>= 1).

    latticeconstant (optional): The lattice constant. If not given,
    then it is extracted form ase.data.
    