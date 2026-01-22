import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
def check_cuts_valid_on_facet_containing_optimal(self, m):
    cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
    self.assertGreaterEqual(len(cuts), 1)
    m.x.fix(5)
    m.y.fix(3)
    m.upper_circle.indicator_var.fix(False)
    m.lower_circle.indicator_var.fix(True)
    m.upper_circle2.indicator_var.fix(False)
    m.lower_circle2.indicator_var.fix(True)
    for i in range(len(cuts)):
        self.assertGreaterEqual(value(cuts[i].body), 0)