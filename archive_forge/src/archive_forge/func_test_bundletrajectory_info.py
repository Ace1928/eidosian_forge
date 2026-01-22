import pytest
import numpy as np
import sys
from subprocess import check_call, check_output
from pathlib import Path
from ase.build import bulk
from ase.io import read, write
from ase.io.pickletrajectory import PickleTrajectory
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io.bundletrajectory import (BundleTrajectory,
def test_bundletrajectory_info(images, bundletraj, capsys):
    print_bundletrajectory_info(bundletraj)
    output, _ = capsys.readouterr()
    natoms = len(images[0])
    expected_substring = f'Number of atoms: {natoms}'
    assert expected_substring in output
    output2 = check_output([sys.executable, '-m', 'ase.io.bundletrajectory', bundletraj], encoding='ascii')
    assert expected_substring in output2