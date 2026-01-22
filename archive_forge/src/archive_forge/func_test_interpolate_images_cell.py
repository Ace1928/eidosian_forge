from ase import Atoms
from ase.neb import interpolate
from ase.constraints import FixAtoms
import numpy as np
import pytest
def test_interpolate_images_cell(images, initial, average_pos):
    interpolate(images, interpolate_cell=True)
    assert images[1].positions == pytest.approx(average_pos)
    assert_interpolated([image.positions for image in images])
    assert_interpolated([image.cell for image in images])