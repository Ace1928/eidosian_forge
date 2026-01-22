import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.data import chemical_symbols
from ase.utils import reader, writer
from .utils import verify_cell_for_export, verify_dictionary
@writer
def write_mustem(fd, *args, **kwargs):
    """Write muSTEM input file.

    Parameters:

    atoms: Atoms object

    keV: float
        Energy of the electron beam in keV required for the image simulation.

    debye_waller_factors: float or dictionary of float with atom type as key
        Debye-Waller factor of each atoms. Since the prismatic/computem
        software use root means square RMS) displacements, the Debye-Waller
        factors (B) needs to be provided in Å² and these values are converted
        to RMS displacement by:

        .. math::

            RMS = \\frac{B}{8\\pi^2}

    occupancies: float or dictionary of float with atom type as key (optional)
        Occupancy of each atoms. Default value is `1.0`.

    comment: str (optional)
        Comments to be written in the first line of the file. If not
        provided, write the total number of atoms and the chemical formula.

    fit_cell_to_atoms: bool (optional)
        If `True`, fit the cell to the atoms positions. If negative coordinates
        are present in the cell, the atoms are translated, so that all
        positions are positive. If `False` (default), the atoms positions and
        the cell are unchanged.
    """
    writer = XtlmuSTEMWriter(*args, **kwargs)
    writer.write_to_file(fd)