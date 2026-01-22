from io import StringIO
import logging
from os.path import join, normpath
import pickle
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn
def make_infeasible_disjunct_model(self):
    m = ConcreteModel()
    m.x = Var(bounds=(1, 12))
    m.y = Var(bounds=(19, 22))
    m.disjunction = Disjunction(expr=[[m.x >= 3 + m.y, m.y == 19.75], [m.y >= 21 + m.x], [m.x == m.y - 9]])
    return m