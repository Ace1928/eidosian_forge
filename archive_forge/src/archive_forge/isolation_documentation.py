import itertools
import collections
import numpy as np
from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals
Isolates components by dimensionality type.

    Given a k-value cutoff the components (connected clusters) are
    identified.  For each component an Atoms object is created, which contains
    that component only.  The geometry of the resulting Atoms object depends
    on the component dimensionality type:

        0D: The cell is a tight box around the atoms.  pbc=[0, 0, 0].
            The cell has no physical meaning.

        1D: The chain is aligned along the z-axis.  pbc=[0, 0, 1].
            The x and y cell directions have no physical meaning.

        2D: The layer is aligned in the x-y plane.  pbc=[1, 1, 0].
            The z cell direction has no physical meaning.

        3D: The original cell is used. pbc=[1, 1, 1].

    Parameters:

    atoms: ASE atoms object
        The system to analyze.
    kcutoff: float
        The k-value cutoff to use.  Default=None, in which case the
        dimensionality scoring parameter is used to select the cutoff.

    Returns:

    components: dict
        key: the component dimenionalities.
        values: a list of Atoms objects for each dimensionality type.
    