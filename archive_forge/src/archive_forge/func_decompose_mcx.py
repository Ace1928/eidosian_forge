from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def decompose_mcx(control_wires, target_wire, work_wires):
    """Decomposes the multi-controlled PauliX gate"""
    num_work_wires_needed = len(control_wires) - 2
    if len(work_wires) >= num_work_wires_needed:
        return _decompose_mcx_with_many_workers(control_wires, target_wire, work_wires)
    return _decompose_mcx_with_one_worker(control_wires, target_wire, work_wires[0])