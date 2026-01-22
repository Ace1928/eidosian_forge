from pyomo.core import quicksum
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.transformation import TransformationFactory
import pyomo.core.expr as EXPR
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
from pyomo.common.config import ConfigDict, ConfigValue
Apply the transformation.