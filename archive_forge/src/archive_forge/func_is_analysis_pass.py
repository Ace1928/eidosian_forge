from __future__ import annotations
import abc
from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable
from inspect import signature
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.base_tasks import GenericPass, PassManagerIR
from qiskit.passmanager.compilation_status import PropertySet, RunState, PassManagerState
from .exceptions import TranspilerError
from .layout import TranspileLayout
@property
def is_analysis_pass(self):
    """Check if the pass is an analysis pass.

        If the pass is an AnalysisPass, that means that the pass can analyze the DAG and write
        the results of that analysis in the property set. Modifications on the DAG are not allowed
        by this kind of pass.
        """
    return isinstance(self, AnalysisPass)