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
def test_mappings_between_original_and_transformed_components(self):
    m = self.make_model()
    mbm = TransformationFactory('gdp.mbigm')
    mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=False)
    d1_block = m.d1.transformation_block
    self.assertIs(mbm.get_src_disjunct(d1_block), m.d1)
    d2_block = m.d2.transformation_block
    self.assertIs(mbm.get_src_disjunct(d2_block), m.d2)
    d3_block = m.d3.transformation_block
    self.assertIs(mbm.get_src_disjunct(d3_block), m.d3)
    for disj in [m.d1, m.d2, m.d3]:
        for comp in ['x1_bounds', 'x2_bounds', 'func']:
            original_cons = disj.component(comp)
            transformed = mbm.get_transformed_constraints(original_cons)
            for cons in transformed:
                self.assertIn(original_cons, mbm.get_src_constraints(cons))