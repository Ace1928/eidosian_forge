import pytest
import numpy as np
from ase.build import bulk
def get_spos(atoms):
    return atoms.get_scaled_positions(wrap=False)