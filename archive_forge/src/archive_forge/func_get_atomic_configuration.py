import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
@classmethod
def get_atomic_configuration(cls, atoms, elements=None, eps=0.04):
    """Returns the atomic configuration of the particle as a list of
        lists. Each list contain the indices of the atoms sitting at the
        same distance from the geometrical center of the particle. Highly
        symmetrical particles will often have many atoms in each shell.

        For further elaboration see:
        J. Montejano-Carrizales and J. Moran-Lopez, Geometrical
        characteristics of compact nanoclusters, Nanostruct. Mater., 1,
        5, 397-409 (1992)

        Parameters:

        elements: Only take into account the elements specified in this
            list. Default is to take all elements into account.

        eps: The distance allowed to separate elements within each shell."""
    atoms = atoms.copy()
    if elements is None:
        e = list(set(atoms.get_chemical_symbols()))
    else:
        e = elements
    atoms.set_constraint()
    atoms.center()
    geo_mid = np.array([(atoms.get_cell() / 2.0)[i][i] for i in range(3)])
    dists = [(np.linalg.norm(geo_mid - atoms[i].position), i) for i in range(len(atoms))]
    dists.sort(key=itemgetter(0))
    atomic_conf = []
    old_dist = -10.0
    for dist, i in dists:
        if abs(dist - old_dist) > eps:
            atomic_conf.append([i])
        else:
            atomic_conf[-1].append(i)
        old_dist = dist
    sorted_elems = sorted(set(atoms.get_chemical_symbols()))
    if e is not None and sorted(e) != sorted_elems:
        for shell in atomic_conf:
            torem = []
            for i in shell:
                if atoms[i].symbol not in e:
                    torem.append(i)
            for i in torem:
                shell.remove(i)
    return atomic_conf