from __future__ import annotations
import itertools
from warnings import warn
import networkx as nx
import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import cite_conventional_cell_algo
from pymatgen.symmetry.kpath import KPathBase, KPathLatimerMunro, KPathSeek, KPathSetyawanCurtarolo
@property
def equiv_labels(self):
    """
        Returns:
            The correspondence between the kpoint symbols in the Latimer and
            Munro convention, Setyawan and Curtarolo, and Hinuma
            conventions respectively. Only generated when path_type = 'all'.
        """
    return self._equiv_labels