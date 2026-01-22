import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.core import (
from pyomo.core.base import TransformationFactory, Reference
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import is_child_of, _get_constraint_transBlock, _to_dict
from pyomo.core.util import target_list
from pyomo.network import Port
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref, ReferenceType
def get_all_M_values_by_constraint(self, model):
    """Returns a dictionary mapping each constraint to a tuple:
        (lower_M_value, upper_M_value), where either can be None if the
        constraint does not have a lower or upper bound (respectively).

        Parameters
        ----------
        model: A GDP model that has been transformed with BigM
        """
    m_values = {}
    for disj in model.component_data_objects(Disjunct, active=None, descend_into=(Block, Disjunct)):
        transBlock = disj.transformation_block
        if transBlock is not None:
            if hasattr(transBlock, 'bigm_src'):
                for cons in transBlock.bigm_src:
                    m_values[cons] = self.get_M_value(cons)
    return m_values