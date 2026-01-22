from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
from qiskit import pulse
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.controlflow import (
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, QubitProperties
from qiskit.providers import Options
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import (
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.utils import optionals as _optionals
Run on the backend using a simulator.

        This method runs circuit jobs (an individual or a list of :class:`~.QuantumCircuit`
        ) and pulse jobs (an individual or a list of :class:`~.Schedule` or
        :class:`~.ScheduleBlock`) using :class:`~.BasicSimulator` or Aer simulator and returns a
        :class:`~qiskit.providers.Job` object.

        If qiskit-aer is installed, jobs will be run using the ``AerSimulator`` with
        noise model of the backend. Otherwise, jobs will be run using the
        ``BasicSimulator`` simulator without noise.

        Noisy simulations of pulse jobs are not yet supported in :class:`~.GenericBackendV2`.

        Args:
            run_input (QuantumCircuit or Schedule or ScheduleBlock or list): An
                individual or a list of
                :class:`~qiskit.circuit.QuantumCircuit`,
                :class:`~qiskit.pulse.ScheduleBlock`, or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object, then the expectation is that the value
                specified will be used instead of what's set in the options
                object.

        Returns:
            Job: The job object for the run

        Raises:
            QiskitError: If a pulse job is supplied and qiskit_aer is not installed.
        