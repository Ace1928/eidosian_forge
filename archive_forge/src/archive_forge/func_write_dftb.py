import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
@writer
def write_dftb(fileobj, images):
    """Write structure in GEN format (refer to DFTB+ manual).
       Multiple snapshots are not allowed. """
    from ase.io.gen import write_gen
    write_gen(fileobj, images)