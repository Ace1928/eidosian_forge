from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_circuit_serializer_name():
    serializer = cg.CircuitSerializer()
    assert serializer.name == _SERIALIZER_NAME
    assert cg.serialization.circuit_serializer.CIRCUIT_SERIALIZER.name == 'v2_5'