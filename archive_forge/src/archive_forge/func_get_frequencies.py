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
def get_frequencies(self, method='standard', direction='central'):
    """Get vibration frequencies in cm^-1."""
    return self.get_vibrations(method=method, direction=direction).get_frequencies()