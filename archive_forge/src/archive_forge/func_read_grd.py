import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def read_grd(filename):
    """ Reads .grd file

    Notes
    -----
    origin_xyz is offset with half a grid point in all directions to be
        compatible with the cube format
    Periodic systems is not guaranteed to be oriented correctly
    """
    from ase.geometry.cell import cellpar_to_cell
    with open(filename, 'r') as fd:
        lines = fd.readlines()
    cell_data = np.array([float(fld) for fld in lines[2].split()])
    cell = cellpar_to_cell(cell_data)
    grid = [int(fld) + 1 for fld in lines[3].split()]
    data = np.empty(grid)
    origin_data = [int(fld) for fld in lines[4].split()[1:]]
    origin_xyz = cell[0] * (-float(origin_data[0]) - 0.5) / (grid[0] - 1) + cell[1] * (-float(origin_data[2]) - 0.5) / (grid[1] - 1) + cell[2] * (-float(origin_data[4]) - 0.5) / (grid[2] - 1)
    fastest_index = int(lines[4].split()[0])
    assert fastest_index in [1, 3]
    if fastest_index == 3:
        grid[0], grid[1] = (grid[1], grid[0])
    dummy_counter = 5
    for i in range(grid[2]):
        for j in range(grid[1]):
            for k in range(grid[0]):
                if fastest_index == 1:
                    data[k, j, i] = float(lines[dummy_counter])
                elif fastest_index == 3:
                    data[j, k, i] = float(lines[dummy_counter])
                dummy_counter += 1
    return (data, cell, origin_xyz)