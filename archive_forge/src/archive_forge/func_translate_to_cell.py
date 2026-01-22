from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def translate_to_cell(self, w, cell):
    """Translate the w'th Wannier function to specified cell"""
    scaled_c = np.angle(self.Z_dww[:3, w, w]) * self.kptgrid / (2 * pi)
    trans = np.array(cell) - np.floor(scaled_c)
    self.translate(w, trans)