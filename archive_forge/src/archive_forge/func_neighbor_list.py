import numpy as np
import itertools
from scipy import sparse as sp
from scipy.spatial import cKDTree
import scipy.sparse.csgraph as csgraph
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import complete_cell, find_mic, wrap_positions
from ase.geometry import minkowski_reduce
from ase.cell import Cell
def neighbor_list(quantities, a, cutoff, self_interaction=False, max_nbins=1000000.0):
    """Compute a neighbor list for an atomic configuration.

    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Parameters:

    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are:

           * 'i' : first atom index
           * 'j' : second atom index
           * 'd' : absolute distance
           * 'D' : distance vector
           * 'S' : shift vector (number of cell boundaries crossed by the bond
             between atom i and j). With the shift vector S, the
             distances D between atoms can be computed from:
             D = a.positions[j]-a.positions[i]+S.dot(a.cell)
    a: :class:`ase.Atoms`
        Atomic configuration.
    cutoff: float or dict
        Cutoff for neighbor search. It can be:

            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood. See :func:`~ase.neighborlist.natural_cutoffs`
              for an example on how to get such a list.

    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.

    Returns:

    i, j, ...: array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a), but the order of (i,j)
        pairs is not guaranteed.

    Examples:

    Examples assume Atoms object *a* and numpy imported as *np*.

    1. Coordination counting::

        i = neighbor_list('i', a, 1.85)
        coord = np.bincount(i)

    2. Coordination counting with different cutoffs for each pair of species::

        i = neighbor_list('i', a,
                          {('H', 'H'): 1.1, ('C', 'H'): 1.3, ('C', 'C'): 1.85})
        coord = np.bincount(i)

    3. Pair distribution function::

        d = neighbor_list('d', a, 10.00)
        h, bin_edges = np.histogram(d, bins=100)
        pdf = h/(4*np.pi/3*(bin_edges[1:]**3 - bin_edges[:-1]**3)) * a.get_volume()/len(a)

    4. Pair potential::

        i, j, d, D = neighbor_list('ijdD', a, 5.0)
        energy = (-C/d**6).sum()
        pair_forces = (6*C/d**5  * (D/d).T).T
        forces_x = np.bincount(j, weights=pair_forces[:, 0], minlength=len(a)) -                    np.bincount(i, weights=pair_forces[:, 0], minlength=len(a))
        forces_y = np.bincount(j, weights=pair_forces[:, 1], minlength=len(a)) -                    np.bincount(i, weights=pair_forces[:, 1], minlength=len(a))
        forces_z = np.bincount(j, weights=pair_forces[:, 2], minlength=len(a)) -                    np.bincount(i, weights=pair_forces[:, 2], minlength=len(a))

    5. Dynamical matrix for a pair potential stored in a block sparse format::

        from scipy.sparse import bsr_matrix
        i, j, dr, abs_dr = neighbor_list('ijDd', atoms)
        energy = (dr.T / abs_dr).T
        dynmat = -(dde * (energy.reshape(-1, 3, 1) * energy.reshape(-1, 1, 3)).T).T                  -(de / abs_dr * (np.eye(3, dtype=energy.dtype) -                    (energy.reshape(-1, 3, 1) * energy.reshape(-1, 1, 3))).T).T
        dynmat_bsr = bsr_matrix((dynmat, j, first_i), shape=(3*len(a), 3*len(a)))

        dynmat_diag = np.empty((len(a), 3, 3))
        for x in range(3):
            for y in range(3):
                dynmat_diag[:, x, y] = -np.bincount(i, weights=dynmat[:, x, y])

        dynmat_bsr += bsr_matrix((dynmat_diag, np.arange(len(a)),
                                  np.arange(len(a) + 1)),
                                 shape=(3 * len(a), 3 * len(a)))

    """
    return primitive_neighbor_list(quantities, a.pbc, a.get_cell(complete=True), a.positions, cutoff, numbers=a.numbers, self_interaction=self_interaction, max_nbins=max_nbins)