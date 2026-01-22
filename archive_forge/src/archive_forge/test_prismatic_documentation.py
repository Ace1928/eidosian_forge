import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.build import bulk
from .test_mustem import make_STO_atoms
Check missing parameter when writing xyz prismatic file.