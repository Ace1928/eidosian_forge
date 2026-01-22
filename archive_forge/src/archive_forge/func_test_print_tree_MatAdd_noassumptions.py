from sympy.printing.tree import tree
from sympy.testing.pytest import XFAIL
def test_print_tree_MatAdd_noassumptions():
    from sympy.matrices.expressions import MatrixSymbol
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    test_str = 'MatAdd: A + B\n+-MatrixSymbol: A\n| +-Str: A\n| +-Integer: 3\n| +-Integer: 3\n+-MatrixSymbol: B\n  +-Str: B\n  +-Integer: 3\n  +-Integer: 3\n'
    assert tree(A + B, assumptions=False) == test_str