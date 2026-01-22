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
def show_as_force(self, n, scale=0.2, show=True):
    return self.get_vibrations().show_as_force(n, scale=scale, show=show)