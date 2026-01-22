import dataclasses
from typing import Any, Dict, List, Sequence, Set, Type, TypeVar, Union
import numpy as np
import cirq, cirq_google
from cirq import _compat, devices
from cirq.devices import noise_utils
from cirq.transformers.heuristic_decompositions import gate_tabulation_math_utils
def build_noise_models(self) -> List['cirq.NoiseModel']:
    """Construct all NoiseModels associated with NoiseProperties."""
    noise_models = super().build_noise_models()
    if self.fsim_errors:
        fsim_ops = {op_id: gate.on(*op_id.qubits) for op_id, gate in self.fsim_errors.items()}
        noise_models.insert(1, devices.InsertionNoiseModel(ops_added=fsim_ops))
    return noise_models