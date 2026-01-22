from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_repr_latex():
    testmat = DenseMatrix([[0, 2]])
    init_printing(True)
    latex_string = testmat._repr_latex_()
    assert isinstance(latex_string, str)
    init_printing(False)