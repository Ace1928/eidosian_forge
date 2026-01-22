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
@pytest.mark.xfail(reason='bug: writes initial magmoms but reads magmoms as part of calculator')
def test_write_read_bundle(images, bundletraj):
    images1 = read(bundletraj, ':')
    assert_images_equal(images, images1)