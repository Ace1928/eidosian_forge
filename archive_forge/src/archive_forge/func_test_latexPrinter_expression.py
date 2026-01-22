import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_expression(self):
    m = generate_model()
    m.express = pyo.Expression(expr=m.x + m.y)
    pstr = latex_printer(m.express)
    bstr = dedent('\n        \\begin{equation} \n             x + y \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)