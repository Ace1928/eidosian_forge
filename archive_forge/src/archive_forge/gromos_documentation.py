import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.utils import reader, writer
Write gromos geometry files (.g96).
    Writes:
    atom positions,
    and simulation cell (if present)
    