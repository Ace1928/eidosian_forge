import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from qiskit.assembler.run_config import RunConfig
from qiskit.assembler.assemble_schedules import _assemble_instructions as _assemble_schedule
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classicalregister import Clbit
from qiskit.exceptions import QiskitError
from qiskit.qobj import (
from qiskit.utils.parallel import parallel_map
def collect_common_calibrations() -> List[GateCalibration]:
    """If a gate calibration appears in all experiments, collect it."""
    common_calibrations = []
    for _, exps_w_cal in exp_indices.items():
        if len(exps_w_cal) == len(experiments):
            _, gate_cal = exps_w_cal[0]
            common_calibrations.append(gate_cal)
    return common_calibrations