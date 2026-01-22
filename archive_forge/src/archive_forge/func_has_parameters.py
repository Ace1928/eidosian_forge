import collections
import copy
import itertools
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagdependency import DAGDependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.utils import optionals as _optionals
def has_parameters(self):
    """Ensure that the template does not have parameters."""
    for node in self.template_dag_dep.get_nodes():
        for param in node.op.params:
            if isinstance(param, ParameterExpression):
                return True
    return False