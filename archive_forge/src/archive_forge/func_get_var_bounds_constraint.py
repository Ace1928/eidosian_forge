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
def get_var_bounds_constraint(self, v, disjunct=None):
    """
        Returns the IndexedConstraint which sets a disaggregated
        variable to be within its bounds when its Disjunct is active and to
        be 0 otherwise. (It is always an IndexedConstraint because each
        bound becomes a separate constraint.)

        Parameters
        ----------
        v: a Var that was created by the hull transformation as a
           disaggregated variable (and so appears on a transformation
           block of some Disjunct)
        disjunct: (For nested Disjunctions) Which Disjunct in the
           hierarchy the bounds Constraint should correspond to.
           Optional since for non-nested models this can be inferred.
        """
    info = v.parent_block().private_data()
    if v in info.bigm_constraint_map:
        if len(info.bigm_constraint_map[v]) == 1:
            return list(info.bigm_constraint_map[v].values())[0]
        elif disjunct is not None:
            return info.bigm_constraint_map[v][disjunct]
        else:
            raise ValueError("It appears that the variable '%s' appears within a nested GDP hierarchy, and no 'disjunct' argument was specified. Please specify for which Disjunct the bounds constraint for '%s' should be returned." % (v, v))
    raise GDP_Error("Either '%s' is not a disaggregated variable, or the disjunction that disaggregates it has not been properly transformed." % v.name)