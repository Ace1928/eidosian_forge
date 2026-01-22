import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.utils import reader
from .utils import verify_cell_for_export, verify_dictionary
@reader
def read_prismatic(fd):
    """Import prismatic and computem xyz input file as an Atoms object.

    Reads cell, atom positions, occupancies and Debye Waller factor.
    The occupancy values and the Debye Waller factors are obtained using the
    `get_array` method and the `occupancies` and `debye_waller_factors` keys,
    respectively. The root means square (RMS) values from the
    prismatic/computem xyz file are converted to Debye-Waller factors (B) in Å²
    by:

    .. math::

        B = RMS^2 * 8\\pi^2

    """
    fd.readline()
    cellpar = [float(i) for i in fd.readline().split()]
    read_data = np.genfromtxt(fname=fd, skip_footer=1)
    RMS = read_data[:, 5] ** 2 * 8 * np.pi ** 2
    atoms = Atoms(symbols=read_data[:, 0], positions=read_data[:, 1:4], cell=cellpar)
    atoms.set_array('occupancies', read_data[:, 4])
    atoms.set_array('debye_waller_factors', RMS)
    return atoms