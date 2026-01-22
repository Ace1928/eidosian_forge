from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
import re
import pytest
from ase.cli.template import prec_round, slice_split, \
from ase.io import read
def test_twoFiles_falseCalc_multipleImages(cli, traj):
    cli.ase('diff', f'{traj}@:2', f'{traj}@2:4')