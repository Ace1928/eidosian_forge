from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_measurement_gate_deserialize() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q) ** 0.5, cirq.measure(q))
    msg = cg.CIRCUIT_SERIALIZER.serialize(circuit)
    assert cg.CIRCUIT_SERIALIZER.deserialize(msg) == circuit