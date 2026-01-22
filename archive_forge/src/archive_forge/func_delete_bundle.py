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
@classmethod
def delete_bundle(cls, filename):
    """Deletes a bundle."""
    if world.rank == 0:
        if not cls.is_bundle(filename, allowempty=True):
            raise IOError('Cannot remove "%s" as it is not a bundle trajectory.' % (filename,))
        if os.path.islink(filename):
            os.remove(filename)
        else:
            shutil.rmtree(filename)
    else:
        while os.path.exists(filename):
            time.sleep(1)
    barrier()