import numpy as np
from collections import namedtuple
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality import topology_scaling
from ase.geometry.dimensionality.bond_generator import next_bond
Performs a k-interval analysis.

    In each k-interval the components (connected clusters) are identified.
    The intervals are sorted according to the scoring parameter, from high
    to low.

    Parameters:

    atoms: ASE atoms object
        The system to analyze. The periodic boundary conditions determine
        the maximum achievable component dimensionality, i.e. pbc=[1,1,0] sets
        a maximum dimensionality of 2.
    method: string
        Analysis method to use, either 'RDA' (default option) or 'TSA'.
        These correspond to the Rank Determination Algorithm of Mounet et al.
        and the Topological Scaling Algorithm (TSA) of Ashton et al.
    merge: boolean
        Decides if k-intervals of the same type (e.g. 01D or 3D) should be
        merged.  Default: true

    Returns:

    intervals: list
        List of KIntervals for each interval identified.  A KInterval is a
        namedtuple with the following field names:

        score: float
            Dimensionality score in the range [0, 1]
        a: float
            The start of the k-interval
        b: float
            The end of the k-interval
        dimtype: str
            The dimensionality type
        h: tuple
            The histogram of the number of components of each dimensionality.
            For example, (8, 0, 3, 0) means eight 0D and three 2D components.
        components: array
            The component ID of each atom.
        cdim: dict
            The component dimensionalities
    