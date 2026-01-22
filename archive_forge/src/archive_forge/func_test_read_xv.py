from io import StringIO
from pathlib import Path
import numpy as np
import pytest
from ase.io import read
from ase.io.siesta import read_struct_out, read_fdf
from ase.units import Bohr
def test_read_xv():
    path = Path('tmp.XV')
    path.write_text(xv_file)
    atoms = read(path)
    assert str(atoms.symbols) == 'Ti2'
    pos = atoms.positions
    assert pos[0] == pytest.approx(0)
    assert pos[1] / Bohr == pytest.approx([0, 3.2, 4.4])
    assert all(atoms.pbc)
    assert atoms.cell / Bohr == pytest.approx(np.array([[5.6, 0, 0], [-2.8, 4.8, 0], [0, 0, 8.9]]))