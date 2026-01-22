import re
import numpy as np
from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from .vasp_parsers import vasp_outcar_parsers as vop
from pathlib import Path
def iread_vasp_out(filename, index=-1):
    """Import OUTCAR type file, as a generator."""
    it = ImageIterator(vop.outcarchunks)
    return it(filename, index=index)