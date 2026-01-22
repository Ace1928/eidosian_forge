from math import pi, sqrt, log
import sys
import numpy as np
from pathlib import Path
import ase.units as units
import ase.io
from ase.parallel import world, paropen
from ase.utils.filecache import get_json_cache
from .data import VibrationsData
from collections import namedtuple
def write_jmol(self):
    """Writes file for viewing of the modes with jmol."""
    with open(self.name + '.xyz', 'w') as fd:
        self._write_jmol(fd)