from typing import TYPE_CHECKING, cast, FrozenSet, Iterable, Mapping, Optional, Tuple
import networkx as nx
from cirq import value
from cirq.devices import device
Get a dictionary mapping from gate family to duration for gates.

        To look up the duration of a specific gate instance / gate type / operation which is part of
        the device's gateset, you can search for its corresponding GateFamily. For example:

        >>> gateset = cirq.Gateset(cirq.ZPowGate)
        >>> durations = {cirq.GateFamily(cirq.ZPowGate): cirq.Duration(nanos=1)}
        >>> grid_device_metadata = cirq.GridDeviceMetadata((), gateset, durations)
        >>>
        >>> my_gate = cirq.Z
        >>> gate_durations = grid_device_metadata.gate_durations
        >>> gate_duration = None
        >>> for gate_family in gate_durations:
        ...     if my_gate in gate_family:
        ...         gate_duration = gate_durations[gate_family]
        ...
        >>> print(gate_duration)
        1 ns
        