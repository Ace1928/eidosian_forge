import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
def check_cuts_valid_at_extreme_pts(self, m):
    extreme_points = [(1, 0, 2, 10), (1, 0, 0, 10), (1, 0, 0, 7), (1, 0, 2, 7), (0, 1, 8, 0), (0, 1, 8, 3), (0, 1, 10, 0), (0, 1, 10, 3)]
    cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
    for cut in cuts.values():
        cut_expr = cut.body
        lower = cut.lower
        upper = cut.upper
        for pt in extreme_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.binary_indicator_var.fix(pt[0])
            m.disjunct2.binary_indicator_var.fix(pt[1])
            check_validity(self, cut_expr, lower, upper)