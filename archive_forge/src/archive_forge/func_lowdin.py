from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def lowdin(U, S=None):
    """Orthonormalize columns of U according to the Lowdin procedure.

    If the overlap matrix is know, it can be specified in S.
    """
    if S is None:
        S = np.dot(dag(U), U)
    eig, rot = np.linalg.eigh(S)
    rot = np.dot(rot / np.sqrt(eig), dag(rot))
    U[:] = np.dot(U, rot)