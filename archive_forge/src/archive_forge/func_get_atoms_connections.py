import os
import time
import math
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from ase.io import write, read
from ase.geometry.cell import cell_to_cellpar
from ase.data import covalent_radii
from ase.ga import get_neighbor_list
def get_atoms_connections(atoms, max_conn=5, no_count_types=None):
    """This method returns a list of the numbers of atoms
    with X number of neighbors. The method utilizes the
    neighbor list and hence inherit the restrictions for
    neighbors. Option added to remove connections between
    defined atom types.
    """
    conn_index = get_connections_index(atoms, max_conn=max_conn, no_count_types=no_count_types)
    no_of_conn = [0] * max_conn
    for i in conn_index:
        no_of_conn[i] += len(conn_index[i])
    return no_of_conn