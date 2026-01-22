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
def instruction_supported(self, operation_name=None, qargs=None, operation_class=None, parameters=None):
    """Return whether the instruction (operation + qubits) is supported by the target

        Args:
            operation_name (str): The name of the operation for the instruction. Either
                this or ``operation_class`` must be specified, if both are specified
                ``operation_class`` will take priority and this argument will be ignored.
            qargs (tuple): The tuple of qubit indices for the instruction. If this is
                not specified then this method will return ``True`` if the specified
                operation is supported on any qubits. The typical application will
                always have this set (otherwise it's the same as just checking if the
                target contains the operation). Normally you would not set this argument
                if you wanted to check more generally that the target supports an operation
                with the ``parameters`` on any qubits.
            operation_class (Type[qiskit.circuit.Instruction]): The operation class to check whether
                the target supports a particular operation by class rather
                than by name. This lookup is more expensive as it needs to
                iterate over all operations in the target instead of just a
                single lookup. If this is specified it will supersede the
                ``operation_name`` argument. The typical use case for this
                operation is to check whether a specific variant of an operation
                is supported on the backend. For example, if you wanted to
                check whether a :class:`~.RXGate` was supported on a specific
                qubit with a fixed angle. That fixed angle variant will
                typically have a name different from the object's
                :attr:`~.Instruction.name` attribute (``"rx"``) in the target.
                This can be used to check if any instances of the class are
                available in such a case.
            parameters (list): A list of parameters to check if the target
                supports them on the specified qubits. If the instruction
                supports the parameter values specified in the list on the
                operation and qargs specified this will return ``True`` but
                if the parameters are not supported on the specified
                instruction it will return ``False``. If this argument is not
                specified this method will return ``True`` if the instruction
                is supported independent of the instruction parameters. If
                specified with any :class:`~.Parameter` objects in the list,
                that entry will be treated as supporting any value, however parameter names
                will not be checked (for example if an operation in the target
                is listed as parameterized with ``"theta"`` and ``"phi"`` is
                passed into this function that will return ``True``). For
                example, if called with::

                    parameters = [Parameter("theta")]
                    target.instruction_supported("rx", (0,), parameters=parameters)

                will return ``True`` if an :class:`~.RXGate` is supported on qubit 0
                that will accept any parameter. If you need to check for a fixed numeric
                value parameter this argument is typically paired with the ``operation_class``
                argument. For example::

                    target.instruction_supported("rx", (0,), RXGate, parameters=[pi / 4])

                will return ``True`` if an RXGate(pi/4) exists on qubit 0.

        Returns:
            bool: Returns ``True`` if the instruction is supported and ``False`` if it isn't.

        """

    def check_obj_params(parameters, obj):
        for index, param in enumerate(parameters):
            if isinstance(param, Parameter) and (not isinstance(obj.params[index], Parameter)):
                return False
            if param != obj.params[index] and (not isinstance(obj.params[index], Parameter)):
                return False
        return True
    if self.num_qubits is None:
        qargs = None
    if qargs is not None:
        qargs = tuple(qargs)
    if operation_class is not None:
        for op_name, obj in self._gate_name_map.items():
            if inspect.isclass(obj):
                if obj != operation_class:
                    continue
                if qargs is None:
                    return True
                elif all((qarg <= self.num_qubits for qarg in qargs)) and len(set(qargs)) == len(qargs):
                    return True
                else:
                    return False
            if isinstance(obj, operation_class):
                if parameters is not None:
                    if len(parameters) != len(obj.params):
                        continue
                    if not check_obj_params(parameters, obj):
                        continue
                if qargs is None:
                    return True
                if qargs in self._gate_map[op_name]:
                    return True
                if self._gate_map[op_name] is None or None in self._gate_map[op_name]:
                    return self._gate_name_map[op_name].num_qubits == len(qargs) and all((x < self.num_qubits for x in qargs))
        return False
    if operation_name in self._gate_map:
        if parameters is not None:
            obj = self._gate_name_map[operation_name]
            if inspect.isclass(obj):
                if qargs is None:
                    return True
                elif all((qarg <= self.num_qubits for qarg in qargs)) and len(set(qargs)) == len(qargs):
                    return True
                else:
                    return False
            if len(parameters) != len(obj.params):
                return False
            for index, param in enumerate(parameters):
                matching_param = False
                if isinstance(obj.params[index], Parameter):
                    matching_param = True
                elif param == obj.params[index]:
                    matching_param = True
                if not matching_param:
                    return False
            return True
        if qargs is None:
            return True
        if qargs in self._gate_map[operation_name]:
            return True
        if self._gate_map[operation_name] is None or None in self._gate_map[operation_name]:
            obj = self._gate_name_map[operation_name]
            if inspect.isclass(obj):
                if qargs is None:
                    return True
                elif all((qarg <= self.num_qubits for qarg in qargs)) and len(set(qargs)) == len(qargs):
                    return True
                else:
                    return False
            else:
                return self._gate_name_map[operation_name].num_qubits == len(qargs) and all((x < self.num_qubits for x in qargs))
    return False