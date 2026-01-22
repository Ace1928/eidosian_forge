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
def get_Ms(self, m):
    return {(m.d1.x1_bounds, m.d2): (0.15, 1), (m.d1.x2_bounds, m.d2): (2.25, 7), (m.d1.x1_bounds, m.d3): (1.5, 8), (m.d1.x2_bounds, m.d3): (-0.2, -2), (m.d2.x1_bounds, m.d1): (-0.15, -1), (m.d2.x2_bounds, m.d1): (-2.25, -7), (m.d2.x1_bounds, m.d3): (1.35, 7), (m.d2.x2_bounds, m.d3): (-2.45, -9), (m.d3.x1_bounds, m.d1): (-1.5, -8), (m.d3.x2_bounds, m.d1): (0.2, 2), (m.d3.x1_bounds, m.d2): (-1.35, -7), (m.d3.x2_bounds, m.d2): (2.45, 9), (m.d1.func, m.d2): (-40, -16.65), (m.d1.func, m.d3): (6.3, 9), (m.d2.func, m.d1): (9.75, 18), (m.d2.func, m.d3): (16.95, 29), (m.d3.func, m.d1): (-21, -7.5), (m.d3.func, m.d2): (-103, -37.65)}