from sympy.categories.diagram_drawing import _GrowableGrid, ArrowStringDescription
from sympy.categories import (DiagramGrid, Object, NamedMorphism,
from sympy.sets.sets import FiniteSet
def test_DiagramGrid_pseudopod():
    A = Object('A')
    B = Object('B')
    C = Object('C')
    D = Object('D')
    E = Object('E')
    F = Object('F')
    A_ = Object("A'")
    B_ = Object("B'")
    C_ = Object("C'")
    D_ = Object("D'")
    E_ = Object("E'")
    f1 = NamedMorphism(A, B, 'f1')
    f2 = NamedMorphism(A, C, 'f2')
    f3 = NamedMorphism(A, D, 'f3')
    f4 = NamedMorphism(A, E, 'f4')
    f5 = NamedMorphism(A, A_, 'f5')
    f6 = NamedMorphism(A, B_, 'f6')
    f7 = NamedMorphism(A, C_, 'f7')
    f8 = NamedMorphism(A, D_, 'f8')
    f9 = NamedMorphism(A, E_, 'f9')
    f10 = NamedMorphism(A, F, 'f10')
    d = Diagram([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
    grid = DiagramGrid(d)
    assert grid.width == 5
    assert grid.height == 3
    assert grid[0, 0] == E
    assert grid[0, 1] == C
    assert grid[0, 2] == C_
    assert grid[0, 3] == E_
    assert grid[0, 4] == F
    assert grid[1, 0] == D
    assert grid[1, 1] == A
    assert grid[1, 2] == A_
    assert grid[1, 3] is None
    assert grid[1, 4] is None
    assert grid[2, 0] == D_
    assert grid[2, 1] == B
    assert grid[2, 2] == B_
    assert grid[2, 3] is None
    assert grid[2, 4] is None
    morphisms = {}
    for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]:
        morphisms[f] = FiniteSet()
    assert grid.morphisms == morphisms