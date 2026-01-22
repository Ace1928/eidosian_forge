from typing import Callable
from cirq import circuits, devices, ops
def linearize_circuit_qubits(circuit: circuits.Circuit, qubit_order: ops.QubitOrderOrList=ops.QubitOrder.DEFAULT) -> None:
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
    qubit_map = {q: devices.LineQubit(i) for i, q in enumerate(qubits)}
    QubitMapper(qubit_map.__getitem__).optimize_circuit(circuit)