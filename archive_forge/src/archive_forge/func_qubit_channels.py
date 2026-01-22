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
def qubit_channels(qubit: int) -> set[chans.Channel]:
    """Returns the set of channels associated with a qubit.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            print(pulse.qubit_channels(0))

    .. parsed-literal::

       {MeasureChannel(0), ControlChannel(0), DriveChannel(0), AcquireChannel(0), ControlChannel(1)}

    .. note:: Requires the active builder context to have a backend set.

    .. note:: A channel may still be associated with another qubit in this list
        such as in the case where significant crosstalk exists.

    """

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
    if isinstance(active_backend(), BackendV2):
        return set(get_qubit_channels_v2(active_backend(), qubit))
    return set(active_backend().configuration().get_qubit_channels(qubit))