from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_ex_633_trespalacios(self):
    exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
    m = exfile.build_simple_nonconvex_gdp()
    self.assertTrue(satisfiable(m) is not False)