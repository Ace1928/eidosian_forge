from __future__ import annotations
import contextvars
import functools
import itertools
import sys
import uuid
import warnings
from collections.abc import Generator, Callable, Iterable
from contextlib import contextmanager
from functools import singledispatchmethod
from typing import TypeVar, ContextManager, TypedDict, Union, Optional, Dict
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import (
from qiskit.providers.backend import BackendV2
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
def get_qubit_channels_v2(backend: BackendV2, qubit: int):
    """Return a list of channels which operate on the given ``qubit``.
        Returns:
            List of ``Channel``\\s operated on my the given ``qubit``.
        """
    channels = []
    for node_qubits in backend.coupling_map:
        if qubit in node_qubits:
            control_channel = backend.control_channel(node_qubits)
            if control_channel:
                channels.extend(control_channel)
    channels.append(backend.drive_channel(qubit))
    channels.append(backend.measure_channel(qubit))
    channels.append(backend.acquire_channel(qubit))
    return channels