import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.table import CustomCell, Table
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
def test_table_cells():
    fig, ax = plt.subplots()
    table = Table(ax)
    cell = table.add_cell(1, 2, 1, 1)
    assert isinstance(cell, CustomCell)
    assert cell is table[1, 2]
    cell2 = CustomCell((0, 0), 1, 2, visible_edges=None)
    table[2, 1] = cell2
    assert table[2, 1] is cell2
    table.properties()
    plt.setp(table)