import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_constraint(self):
    m = generate_model()
    pstr = latex_printer(m.constraint_1)
    bstr = dedent('\n        \\begin{equation} \n             x^{2} + y^{-2} - x y z + 1 = 2 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)