import logging
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.trustregion.util import minIgnoreNone, maxIgnoreNone
from pyomo.core import (
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.visitor import identify_variables, ExpressionReplacementVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
from pyomo.opt import SolverFactory, check_optimal_termination
def updateSurrogateModel(self):
    """
        The parameters needed for the surrogate model are the values of:
            b(w_k)      : basis_model_output
            d(w_k)      : truth_model_output
            grad b(w_k) : grad_basis_model_output
            grad d(w_k) : grad_truth_model_output
        """
    b = self.data
    for i, y in b.ef_outputs.items():
        b.basis_model_output[i] = value(b.basis_expressions[y])
        b.truth_model_output[i] = value(b.truth_models[y])
        gradBasis = differentiate(b.basis_expressions[y], wrt_list=b.ef_inputs[i])
        gradTruth = differentiate(b.truth_models[y], wrt_list=b.ef_inputs[i])
        for j, w in enumerate(b.ef_inputs[i]):
            b.grad_basis_model_output[i, j] = gradBasis[j]
            b.grad_truth_model_output[i, j] = gradTruth[j]
            b.value_of_ef_inputs[i, j] = value(w)