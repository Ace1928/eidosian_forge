from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
import re
import pytest
from ase.cli.template import prec_round, slice_split, \
from ase.io import read
def test_twoFiles_trueCalc_multipleImages(cli, traj):
    stdout = cli.ase('diff', f'{traj}@:2', f'{traj}@2:4', '-c', '--rank-order', 'dfx', '--as-csv')
    stdout = [row.split(',') for row in stdout.split('\n')]
    stdout = [row for row in stdout if len(row) > 4]
    header = stdout[0]
    body = stdout[1:len(stdout) // 2 - 1]
    for c in range(len(header)):
        if header[c] == 'Δfx':
            break
    dfx_ordered = [float(row[c]) for row in body]
    for i in range(len(dfx_ordered) - 2):
        assert dfx_ordered[i] <= dfx_ordered[i + 1]