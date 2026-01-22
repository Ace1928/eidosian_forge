from sympy.categories.diagram_drawing import _GrowableGrid, ArrowStringDescription
from sympy.categories import (DiagramGrid, Object, NamedMorphism,
from sympy.sets.sets import FiniteSet
def test_GrowableGrid():
    grid = _GrowableGrid(1, 2)
    assert grid.width == 1
    assert grid.height == 2
    assert grid[0, 0] is None
    assert grid[1, 0] is None
    grid[0, 0] = 1
    grid[1, 0] = 'two'
    assert grid[0, 0] == 1
    assert grid[1, 0] == 'two'
    grid.append_row()
    assert grid.width == 1
    assert grid.height == 3
    assert grid[0, 0] == 1
    assert grid[1, 0] == 'two'
    assert grid[2, 0] is None
    grid.append_column()
    assert grid.width == 2
    assert grid.height == 3
    assert grid[0, 0] == 1
    assert grid[1, 0] == 'two'
    assert grid[2, 0] is None
    assert grid[0, 1] is None
    assert grid[1, 1] is None
    assert grid[2, 1] is None
    grid = _GrowableGrid(1, 2)
    grid[0, 0] = 1
    grid[1, 0] = 'two'
    grid.prepend_row()
    assert grid.width == 1
    assert grid.height == 3
    assert grid[0, 0] is None
    assert grid[1, 0] == 1
    assert grid[2, 0] == 'two'
    grid.prepend_column()
    assert grid.width == 2
    assert grid.height == 3
    assert grid[0, 0] is None
    assert grid[1, 0] is None
    assert grid[2, 0] is None
    assert grid[0, 1] is None
    assert grid[1, 1] == 1
    assert grid[2, 1] == 'two'