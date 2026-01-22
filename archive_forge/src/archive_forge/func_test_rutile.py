import pytest
import numpy as np
from ase import io
def test_rutile():
    fname = 'TiO2_rutile.cml'
    with open(fname, 'w') as fd:
        fd.write(tio2)
    atoms = io.read(fname)
    assert atoms.pbc.all()
    cell = atoms.cell
    assert str(atoms.symbols) == 'Ti2O4'
    assert atoms[1].position == pytest.approx(cell.diagonal() / 2)
    assert cell[1, 1] == cell[2, 2]
    assert cell == pytest.approx(np.diag(cell.diagonal()))