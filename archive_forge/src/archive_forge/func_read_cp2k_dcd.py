import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def read_cp2k_dcd(fileobj, index=-1, ref_atoms=None, aligned=False):
    """ Read a DCD file created by CP2K.

    To yield one Atoms object at a time use ``iread_cp2k_dcd``.

    Note: Other programs use other formats, they are probably not compatible.

    If ref_atoms is not given, all atoms will have symbol 'X'.

    To make sure that this is a dcd file generated with the *DCD_ALIGNED_CELL* key
    in the CP2K input, you need to set ``aligned`` to *True* to get cell information.
    Make sure you do not set it otherwise, the cell will not match the atomic coordinates!
    See the following url for details:
    https://groups.google.com/forum/#!searchin/cp2k/Outputting$20cell$20information$20and$20fractional$20coordinates%7Csort:relevance/cp2k/72MhYykrSrQ/5c9Jaw7S9yQJ.
    """
    dtype, natoms, nsteps, header_end = _read_metainfo(fileobj)
    bytes_per_timestep = _bytes_per_timestep(natoms)
    if ref_atoms:
        symbols = ref_atoms.get_chemical_symbols()
    else:
        symbols = ['X' for i in range(natoms)]
    if natoms != len(symbols):
        raise ValueError('Length of ref_atoms does not match natoms from dcd file')
    trbl = index2range(index, nsteps)
    for index in trbl:
        frame_pos = int(header_end + index * bytes_per_timestep)
        fileobj.seek(frame_pos)
        yield _read_cp2k_dcd_frame(fileobj, dtype, natoms, symbols, aligned=aligned)