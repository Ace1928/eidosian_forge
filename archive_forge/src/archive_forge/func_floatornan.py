import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def floatornan(value):
    """Converts the argument into a float if possible, np.nan if not."""
    try:
        output = float(value)
    except ValueError:
        output = np.nan
    return output