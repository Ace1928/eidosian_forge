from a backend
from __future__ import annotations
import itertools
from typing import Optional, List, Any
from collections.abc import Mapping
from collections import defaultdict
import datetime
import io
import logging
import inspect
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.calibration_entries import CalibrationEntry, ScheduleDef
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.pulse.exceptions import PulseError, UnassignedDurationError
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import QubitProperties  # pylint: disable=unused-import
from qiskit.providers.models.backendproperties import BackendProperties
def get_non_global_operation_names(self, strict_direction=False):
    """Return the non-global operation names for the target

        The non-global operations are those in the target which don't apply
        on all qubits (for single qubit operations) or all multi-qubit qargs
        (for multi-qubit operations).

        Args:
            strict_direction (bool): If set to ``True`` the multi-qubit
                operations considered as non-global respect the strict
                direction (or order of qubits in the qargs is significant). For
                example, if ``cx`` is defined on ``(0, 1)`` and ``ecr`` is
                defined over ``(1, 0)`` by default neither would be considered
                non-global, but if ``strict_direction`` is set ``True`` both
                ``cx`` and ``ecr`` would be returned.

        Returns:
            List[str]: A list of operation names for operations that aren't global in this target
        """
    if strict_direction:
        if self._non_global_strict_basis is not None:
            return self._non_global_strict_basis
        search_set = self._qarg_gate_map.keys()
    else:
        if self._non_global_basis is not None:
            return self._non_global_basis
        search_set = {frozenset(qarg) for qarg in self._qarg_gate_map if qarg is not None and len(qarg) != 1}
    incomplete_basis_gates = []
    size_dict = defaultdict(int)
    size_dict[1] = self.num_qubits
    for qarg in search_set:
        if qarg is None or len(qarg) == 1:
            continue
        size_dict[len(qarg)] += 1
    for inst, qargs in self._gate_map.items():
        qarg_sample = next(iter(qargs))
        if qarg_sample is None:
            continue
        if not strict_direction:
            qargs = {frozenset(qarg) for qarg in qargs}
        if len(qargs) != size_dict[len(qarg_sample)]:
            incomplete_basis_gates.append(inst)
    if strict_direction:
        self._non_global_strict_basis = incomplete_basis_gates
    else:
        self._non_global_basis = incomplete_basis_gates
    return incomplete_basis_gates