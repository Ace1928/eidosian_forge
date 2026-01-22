import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_iteratedConstraints(self):
    m = generate_model()
    pstr = latex_printer(m.constraint_7)
    bstr = dedent('\n        \\begin{equation} \n              \\left( x + y \\right)  \\sum_{ i \\in I  } v_{i} + u_{i,j}^{2} \\leq 0  \\qquad \\forall j \\in I \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)
    pstr = latex_printer(m.constraint_8)
    bstr = dedent('\n        \\begin{equation} \n             \\sum_{ i \\in K  } p_{i} = 1 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)