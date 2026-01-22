from ase import Atoms
from ase.neb import interpolate
from ase.constraints import FixAtoms
import numpy as np
import pytest
def test_interpolate_images_fixed(images, initial, average_pos):
    for image in images:
        image.set_constraint(FixAtoms([0]))
    with pytest.raises(RuntimeError, match='Constraint\\(s\\) in image number'):
        interpolate(images)
    interpolate(images, apply_constraint=True)
    assert images[1].positions == pytest.approx(images[0].positions)
    assert np.allclose(images[1].cell, initial.cell)
    interpolate(images, apply_constraint=False)
    assert images[1].positions == pytest.approx(average_pos)
    assert_interpolated([image.positions for image in images])
    assert np.allclose(images[1].cell, initial.cell)