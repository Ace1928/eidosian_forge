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
def remove_common_gate_calibrations(exps: List[QasmQobjExperiment]) -> None:
    """For calibrations that appear in all experiments, remove them from the individual
        experiment's ``config.calibrations``."""
    for _, exps_w_cal in exp_indices.items():
        if len(exps_w_cal) == len(exps):
            for exp_idx, gate_cal in exps_w_cal:
                exps[exp_idx].config.calibrations.gates.remove(gate_cal)