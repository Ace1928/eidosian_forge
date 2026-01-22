import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
def test_relax_integrality(self):
    self.model.d = Var(within=Integers, bounds=(-2, 3))
    instance = self.model.create_instance()
    instance_cloned = instance.clone()
    xfrm = TransformationFactory('core.relax_integer_vars')
    rinst = xfrm.create_using(instance_cloned)
    self.assertEqual(type(rinst.d.domain), RealSet)
    self.assertEqual(rinst.d.bounds, (-2, 3))
    self.assertIs(instance.d.domain, Integers)
    self.assertIs(instance_cloned.d.domain, Integers)