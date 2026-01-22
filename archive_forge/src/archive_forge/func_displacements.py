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
def displacements(self):
    yield self._eq_disp()
    for a, i in self._iter_ai():
        for sign in [-1, 1]:
            for ndisp in range(1, self.nfree // 2 + 1):
                yield self._disp(a, i, sign * ndisp)