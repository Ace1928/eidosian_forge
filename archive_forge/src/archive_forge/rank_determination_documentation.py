import numpy as np
from collections import defaultdict
from ase.geometry.dimensionality.disjoint_set import DisjointSet

        Determines the dimensionality and constituent atoms of each component.

        Returns:
        components: array    The component ID of every atom
        