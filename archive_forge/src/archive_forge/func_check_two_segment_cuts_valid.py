import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
def check_two_segment_cuts_valid(self, m):
    cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
    for cut in cuts.values():
        cut_expr = cut.body
        cut_lower = cut.lower
        cut_upper = cut.upper
        m.x.fix(0)
        m.disj2.indicator_var.fix(False)
        check_validity(self, cut_expr, cut_lower, cut_upper, TOL=1e-08)
        m.x.fix(1)
        check_validity(self, cut_expr, cut_lower, cut_upper)
        m.x.fix(2)
        m.disj2.indicator_var.fix(True)
        check_validity(self, cut_expr, cut_lower, cut_upper)
        m.x.fix(3)
        check_validity(self, cut_expr, cut_lower, cut_upper)