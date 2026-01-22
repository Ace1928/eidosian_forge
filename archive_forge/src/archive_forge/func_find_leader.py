from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def find_leader(self, index):
    """Find in DSU."""
    if index not in self.leader:
        self.leader[index] = index
        self.group[index] = []
        return index
    if self.leader[index] == index:
        return index
    self.leader[index] = self.find_leader(self.leader[index])
    return self.leader[index]