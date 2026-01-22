import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def sub_routine(label_map):
    """Quantum function to initalize state in tests"""
    qml.Hadamard(wires=label_map[0])
    qml.RX(0.12, wires=label_map[1])
    qml.RY(3.45, wires=label_map[2])