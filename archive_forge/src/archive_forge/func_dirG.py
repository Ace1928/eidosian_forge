import os
import time
import subprocess
import re
import warnings
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
def dirG(self, dk, bzone=(0, 0, 0)):
    nx, ny, nz = self['wannier_kpts']
    dx = dk // (ny * nz) + bzone[0] * nx
    dy = dk // nz % ny + bzone[1] * ny
    dz = dk % nz + bzone[2] * nz
    return (dx, dy, dz)