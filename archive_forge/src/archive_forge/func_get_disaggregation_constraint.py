import logging
from collections import defaultdict
from pyomo.common.autoslots import AutoSlots
import pyomo.common.config as cfg
from pyomo.common import deprecated
from pyomo.common.collections import ComponentMap, ComponentSet, DefaultComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import ZeroConstant
import pyomo.core.expr as EXPR
from pyomo.core.base import TransformationFactory, Reference
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import _DisjunctData
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.util.vars_from_expressions import get_vars_from_components
from weakref import ref as weakref_ref
def get_disaggregation_constraint(self, original_var, disjunction, raise_exception=True):
    """
        Returns the disaggregation (re-aggregation?) constraint
        (which links the disaggregated variables to their original)
        corresponding to original_var and the transformation of disjunction.

        Parameters
        ----------
        original_var: a Var which was disaggregated in the transformation
                      of Disjunction disjunction
        disjunction: a transformed Disjunction containing original_var
        """
    for disjunct in disjunction.disjuncts:
        transBlock = disjunct.transformation_block
        if transBlock is not None:
            break
    if transBlock is None:
        raise GDP_Error("Disjunction '%s' has not been properly transformed: None of its disjuncts are transformed." % disjunction.name)
    try:
        cons = transBlock.parent_block()._disaggregationConstraintMap[original_var][disjunction]
    except:
        if raise_exception:
            logger.error("It doesn't appear that '%s' is a variable that was disaggregated by Disjunction '%s'" % (original_var.name, disjunction.name))
            raise
        return None
    while not cons.active:
        cons = self.get_transformed_constraints(cons)[0]
    return cons