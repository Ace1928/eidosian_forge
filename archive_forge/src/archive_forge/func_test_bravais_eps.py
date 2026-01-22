import numpy as np
from ase.cell import Cell
def test_bravais_eps():
    cellpar = np.array([3.42864, 3.42864, 3.42864, 125.788, 125.788, 80.236])
    cell = Cell.new(cellpar)
    mclc = cell.get_bravais_lattice(eps=0.0001)
    bct = cell.get_bravais_lattice(eps=0.001)
    print(mclc)
    print(bct)
    assert mclc.name == 'MCLC'
    assert bct.name == 'BCT'
    perfect_bct_cell = bct.tocell()
    assert perfect_bct_cell.get_bravais_lattice().name == 'BCT'