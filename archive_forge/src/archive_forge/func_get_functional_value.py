from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def get_functional_value(self):
    """Calculate the value of the spread functional.

        ::

          Tr[|ZI|^2]=sum(I)sum(n) w_i|Z_(i)_nn|^2,

        where w_i are weights."""
    a_d = np.sum(np.abs(self.Z_dww.diagonal(0, 1, 2)) ** 2, axis=1)
    return np.dot(a_d, self.weight_d).real