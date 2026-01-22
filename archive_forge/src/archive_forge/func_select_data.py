import os
import sys
import shutil
import time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import jsonio
from ase.io.ulm import open as ulmopen
from ase.parallel import paropen, world, barrier
from ase.calculators.singlepoint import (SinglePointCalculator,
def select_data(self, data, value):
    """Selects if a given data type should be written.

        Data can be written in every frame (specify True), in the
        first frame only (specify 'only') or not at all (specify
        False).  Not all data types support the 'only' keyword, if not
        supported it is interpreted as True.

        The following data types are supported, the letter in parenthesis
        indicates the default:

        positions (T), numbers (O), tags (O), masses (O), momenta (T),
        forces (T), energy (T), energies (F), stress (F), magmoms (T)

        If a given property is not present during the first write, it
        will be not be saved at all.
        """
    if value not in (True, False, 'once'):
        raise ValueError('Unknown write mode')
    if data not in self.datatypes:
        raise ValueError('Unsupported data type: ' + data)
    self.datatypes[data] = value