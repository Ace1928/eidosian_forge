import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_old_json():
    """Older versions of PauliStringPhasor did not have a qubit field."""
    old_json = '\n    {\n      "cirq_type": "PauliStringPhasor",\n      "pauli_string": {\n        "cirq_type": "PauliString",\n        "qubit_pauli_map": [\n          [\n            {\n              "cirq_type": "LineQubit",\n              "x": 0\n            },\n            {\n              "cirq_type": "_PauliX",\n              "exponent": 1.0,\n              "global_shift": 0.0\n            }\n          ],\n          [\n            {\n              "cirq_type": "LineQubit",\n              "x": 1\n            },\n            {\n              "cirq_type": "_PauliY",\n              "exponent": 1.0,\n              "global_shift": 0.0\n            }\n          ],\n          [\n            {\n              "cirq_type": "LineQubit",\n              "x": 2\n            },\n            {\n              "cirq_type": "_PauliZ",\n              "exponent": 1.0,\n              "global_shift": 0.0\n            }\n          ]\n        ],\n        "coefficient": {\n          "cirq_type": "complex",\n          "real": 1.0,\n          "imag": 0.0\n        }\n      },\n      "exponent_neg": 0.2,\n      "exponent_pos": 0.1\n    }\n    '
    phasor = cirq.read_json(json_text=old_json)
    assert phasor == cirq.PauliStringPhasor((1 + 0j) * cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)) * cirq.Z(cirq.LineQubit(2)), qubits=(cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2)), exponent_neg=0.2, exponent_pos=0.1)