import os
from filecmp import cmp
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.collections import OrderedSet
from pyomo.common.fileutils import this_file_dir
import pyomo.core.expr as EXPR
from pyomo.core.base import SymbolMap
from pyomo.environ import (
from pyomo.repn.plugins.baron_writer import expression_to_string
def test_trig_generates_exception(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 2 * 3.1415))
    m.obj = Objective(expr=sin(m.x))
    with self.assertRaisesRegex(RuntimeError, 'The BARON .BAR format does not support the unary function "sin"'):
        test_fname = self._get_fnames()[1]
        self._cleanup(test_fname)
        m.write(test_fname, format='bar')
    self._cleanup(test_fname)