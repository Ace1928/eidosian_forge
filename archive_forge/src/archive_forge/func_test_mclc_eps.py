import numpy as np
from ase.cell import Cell
def test_mclc_eps():
    a = 6.41
    c = 5.87
    alpha = 76.7
    beta = 103.3
    gamma = 152.2
    cell = Cell.new([a, a, c, alpha, beta, gamma])
    lat = cell.get_bravais_lattice(eps=0.01)
    print(lat)
    assert lat.name == 'MCLC'