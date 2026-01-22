from dataclasses import replace
from functools import partial
from typing import Union, Tuple, Sequence
import concurrent.futures
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import (
from pennylane.ops.qubit.observables import BasisStateProjector
from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .default_qubit import accepted_sample_measurement
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
def measure_statistical(self, circuit, stim_circuit, seed=None):
    """Given a circuit, compute samples and return the statistical measurement results."""
    num_shots = circuit.shots.total_shots
    sample_seed = seed if isinstance(seed, int) else self._rng.integers(2 ** 31 - 1, size=1)[0]
    measurement_map = {ExpectationMP: self._sample_expectation, VarianceMP: self._sample_variance, ClassicalShadowMP: self._sample_classical_shadow, ShadowExpvalMP: self._sample_expval_shadow}
    results = []
    for meas in circuit.measurements:
        measurement_func = measurement_map.get(type(meas), None)
        if measurement_func is not None:
            res = measurement_func(meas, stim_circuit, shots=num_shots, seed=sample_seed)
        else:
            meas_wires = meas.wires if meas.wires else range(stim_circuit.num_qubits)
            wire_order = {wire: idx for idx, wire in enumerate(meas.wires)}
            meas_op = meas.obs or qml.prod(*[qml.Z(idx) for idx in meas_wires])
            samples = self._measure_observable_sample(meas_op, stim_circuit, num_shots, sample_seed)[0]
            if len(samples) > 1:
                raise qml.QuantumFunctionError(f'Observable {meas_op.name} is not supported for rotating probabilities on {self.name}.')
            res = meas.process_samples(samples=np.array(samples), wire_order=wire_order)
            if isinstance(meas, CountsMP):
                res = res[0]
            elif isinstance(meas, SampleMP):
                res = np.squeeze(res)
        results.append(res)
    return results