from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.standard_repn import generate_standard_repn
Propagates fixed-to-zero for sums of only positive (or negative) vars.

    If :math:`z` is fixed to zero and :math:`z = x_1 + x_2 + x_3` and
    :math:`x_1`, :math:`x_2`, :math:`x_3` are all non-negative or all
    non-positive, then :math:`x_1`, :math:`x_2`, and :math:`x_3` will be fixed
    to zero.

    