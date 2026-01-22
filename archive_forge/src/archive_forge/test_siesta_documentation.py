from io import StringIO
from pathlib import Path
import numpy as np
import pytest
from ase.io import read
from ase.io.siesta import read_struct_out, read_fdf
from ase.units import Bohr

"Hand-written" dummy file based on HCP Ti
