import pytest
import numpy as np
from ase import Atoms
from ase.build import niggli_reduce
@pytest.mark.parametrize('i', range(len(cells_in)))
def test_niggli(i):
    cell = cells_in[i]
    conf.set_cell(cell)
    niggli_reduce(conf)
    cell = conf.get_cell()
    diff = np.linalg.norm(cell - cells_out[i])
    assert diff < 1e-05, 'Difference between unit cells is too large! ({0})'.format(diff)