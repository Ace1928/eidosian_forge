import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.dependencies import (
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
@unittest.skipUnless(scipy_available, 'scipy is not available.')
@unittest.skipUnless(asl_available, 'pynumero_ASL is not available')
def test_nlp_active_error(self):
    m = pyo.ConcreteModel()
    m.v1 = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.v1 == 1.0)
    m.c2 = pyo.Constraint(expr=m.v1 == 2.0)
    m._obj = pyo.Objective(expr=0.0)
    nlp = PyomoNLP(m)
    with self.assertRaisesRegex(ValueError, 'inactive constraints'):
        igraph = IncidenceGraphInterface(nlp, active=False)