import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def assert_equivalent_to_deferred(circuit: cirq.Circuit):
    qubits = list(circuit.all_qubits())
    sim = cirq.Simulator()
    num_qubits = len(qubits)
    dimensions = [q.dimension for q in qubits]
    for i in range(np.prod(dimensions)):
        bits = cirq.big_endian_int_to_digits(i, base=dimensions)
        modified = cirq.Circuit()
        for j in range(num_qubits):
            modified.append(cirq.XPowGate(dimension=qubits[j].dimension)(qubits[j]) ** bits[j])
        modified.append(circuit)
        deferred = cirq.defer_measurements(modified)
        result = sim.simulate(modified)
        result1 = sim.simulate(deferred)
        np.testing.assert_equal(result.measurements, result1.measurements)