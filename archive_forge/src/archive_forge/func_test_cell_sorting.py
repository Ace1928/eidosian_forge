import unittest
import pytest
from shapely.algorithms.polylabel import Cell, polylabel
from shapely.errors import TopologicalError
from shapely.geometry import LineString, Point, Polygon
def test_cell_sorting(self):
    """
        Tests rich comparison operators of Cells for use in the polylabel
        minimum priority queue.

        """
    polygon = Point(0, 0).buffer(100)
    cell1 = Cell(0, 0, 50, polygon)
    cell2 = Cell(50, 50, 50, polygon)
    assert cell1 < cell2
    assert cell1 <= cell2
    assert (cell2 <= cell1) is False
    assert cell1 == cell1
    assert (cell1 == cell2) is False
    assert cell1 != cell2
    assert (cell1 != cell1) is False
    assert cell2 > cell1
    assert (cell1 > cell2) is False
    assert cell2 >= cell1
    assert (cell1 >= cell2) is False