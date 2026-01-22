from __future__ import annotations
import dataclasses
from typing import Iterable, Tuple, Set, Union, TypeVar, TYPE_CHECKING
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.quantumregister import QuantumRegister
def unify_circuit_resources(circuits: Iterable[QuantumCircuit]) -> Iterable[QuantumCircuit]:
    """
    Ensure that all the given ``circuits`` have all the same qubits, clbits and registers, and
    that they are defined in the same order.  The order is important for binding when the bodies are
    used in the 3-tuple :obj:`.Instruction` context.

    This function will preferentially try to mutate its inputs if they share an ordering, but if
    not, it will rebuild two new circuits.  This is to avoid coupling too tightly to the inner
    class; there is no real support for deleting or re-ordering bits within a :obj:`.QuantumCircuit`
    context, and we don't want to rely on the *current* behaviour of the private APIs, since they
    are very liable to change.  No matter the method used, circuits with unified bits and registers
    are returned.
    """
    circuits = tuple(circuits)
    if len(circuits) < 2:
        return circuits
    qubits = []
    clbits = []
    for circuit in circuits:
        if circuit.qubits[:len(qubits)] != qubits:
            return _unify_circuit_resources_rebuild(circuits)
        if circuit.clbits[:len(qubits)] != clbits:
            return _unify_circuit_resources_rebuild(circuits)
        if circuit.num_qubits > len(qubits):
            qubits = list(circuit.qubits)
        if circuit.num_clbits > len(clbits):
            clbits = list(circuit.clbits)
    for circuit in circuits:
        circuit.add_bits(qubits[circuit.num_qubits:])
        circuit.add_bits(clbits[circuit.num_clbits:])
    return _unify_circuit_registers(circuits)