import numpy as np
import cirq
import pytest
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
def l2norm(result: np.ndarray):
    return np.sum((expected_result - result) ** 2)